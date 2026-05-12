//
// Inherit from SJTU-CV-2021/autoaim/detector/TRTModule.hpp commit 7093b430 Harry-hhj on 21-05-24.
// Modified by Haoran Jiang on 21-10-02: Refact framework.
// Manage TRT Inference
//

#include "TRTModule.hpp"
#include <fstream>
#include <filesystem>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/imgproc.hpp>

#define TRT_ASSERT(expr)                        \
    do                                          \
    {                                           \
        if (!(expr))                            \
        {                                       \
            LOGE_S("[TRT] assert fail: '" #expr "'"); \
            exit(-1);                           \
        }                                       \
    } while (0)

using namespace nvinfer1;

Logger glogger;

static inline size_t get_dims_size(const Dims &dims)
{
    size_t sz = 1;
    for (int i = 0; i < dims.nbDims; i++)
        sz *= dims.d[i];
    return sz;
}

template <class F, class T, class... Ts>
T reduce(F &&func, T x, Ts... xs)
{
    if constexpr (sizeof...(Ts) > 0)
    {
        return func(x, reduce(std::forward<F>(func), xs...));
    }
    else
    {
        return x;
    }
}

template <class T, class... Ts>
T reduce_max(T x, Ts... xs)
{
    return reduce([](auto &&a, auto &&b)
                  { return std::max(a, b); },
                  x, xs...);
}

template <class T, class... Ts>
T reduce_min(T x, Ts... xs)
{
    return reduce([](auto &&a, auto &&b)
                  { return std::min(a, b); },
                  x, xs...);
}

static inline bool is_overlap(const bbox_t &box1, const bbox_t &box2)
{
    cv::Rect2f bbox1, bbox2;
    bbox1.x = reduce_min(box1.pts[0].x, box1.pts[1].x, box1.pts[2].x, box1.pts[3].x);
    bbox1.y = reduce_min(box1.pts[0].y, box1.pts[1].y, box1.pts[2].y, box1.pts[3].y);
    bbox1.width = reduce_max(box1.pts[0].x, box1.pts[1].x, box1.pts[2].x, box1.pts[3].x) - bbox1.x;
    bbox1.height = reduce_max(box1.pts[0].y, box1.pts[1].y, box1.pts[2].y, box1.pts[3].y) - bbox1.y;
    bbox2.x = reduce_min(box2.pts[0].x, box2.pts[1].x, box2.pts[2].x, box2.pts[3].x);
    bbox2.y = reduce_min(box2.pts[0].y, box2.pts[1].y, box2.pts[2].y, box2.pts[3].y);
    bbox2.width = reduce_max(box2.pts[0].x, box2.pts[1].x, box2.pts[2].x, box2.pts[3].x) - bbox2.x;
    bbox2.height = reduce_max(box2.pts[0].y, box2.pts[1].y, box2.pts[2].y, box2.pts[3].y) - bbox2.y;
    return (bbox1 & bbox2).area() > 0;
}

static inline int argmax(const float *ptr, int len)
{
    int max_arg = 0;
    for (int i = 1; i < len; i++)
    {
        if (ptr[i] > ptr[max_arg])
            max_arg = i;
    }
    return max_arg;
}

constexpr float inv_sigmoid(float x)
{
    return -std::log(1 / x - 1);
}

constexpr float sigmoid(float x)
{
    return 1 / (1 + std::exp(-x));
}

constexpr float CONF_THRESH = 0.65f;
constexpr float LOGIT_THRESH = 0.619f;
constexpr float NMS_THRESH = 0.45f;

TRTModule::TRTModule(const std::string &onnx_file) : BackEnd(), runtime(nullptr), engine(nullptr), context(nullptr)
{
    std::filesystem::path onnx_file_path(onnx_file);
    auto cache_file_path = onnx_file_path;
    cache_file_path.replace_extension("cache");
    if (std::filesystem::exists(cache_file_path))
    {
        build_engine_from_cache(cache_file_path.c_str());
    }
    else
    {
        build_engine_from_onnx(onnx_file_path.c_str());
        cache_engine(cache_file_path.c_str());
    }
    TRT_ASSERT((context = engine->createExecutionContext()) != nullptr);
    TRT_ASSERT((input_idx = engine->getBindingIndex("input")) == 0);
    TRT_ASSERT((output_idx = engine->getBindingIndex("output-topk")) == 1);
    auto input_dims = engine->getBindingDimensions(input_idx);
    auto output_dims = engine->getBindingDimensions(output_idx);
    input_sz = get_dims_size(input_dims);
    output_sz = get_dims_size(output_dims);
    TRT_ASSERT(cudaMalloc(&device_buffer[input_idx], input_sz * sizeof(float)) == 0);
    TRT_ASSERT(cudaMalloc(&device_buffer[output_idx], output_sz * sizeof(float)) == 0);
    TRT_ASSERT(cudaStreamCreate(&stream) == 0);
    output_buffer = new float[output_sz];
    TRT_ASSERT(output_buffer != nullptr);
}

TRTModule::~TRTModule()
{
    delete[] output_buffer;
    cudaStreamDestroy(stream);
    cudaFree(device_buffer[output_idx]);
    cudaFree(device_buffer[input_idx]);
    delete context;
    delete engine;
    delete runtime;
}

void TRTModule::build_engine_from_onnx(const std::string &onnx_file)
{
    std::cout << "[INFO]: build engine from onnx" << std::endl;
    auto builder = createInferBuilder(glogger);
    TRT_ASSERT(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    TRT_ASSERT(network != nullptr);
    auto parser = nvonnxparser::createParser(*network, glogger);
    TRT_ASSERT(parser != nullptr);
    parser->parseFromFile(onnx_file.c_str(), static_cast<int>(ILogger::Severity::kINFO));
    auto yolov5_output = network->getOutput(0);
    auto output_dims = yolov5_output->getDimensions();
    TRT_ASSERT(output_dims.nbDims == 3);
    TRT_ASSERT(output_dims.d[2] >= 9);
    const int num_anchors = output_dims.d[1];
    if (output_dims.d[2] == 21)
    {
        yolov5_output->setName("output-topk");
    }
    else
    {
        auto slice_layer = network->addSlice(*yolov5_output, Dims3{0, 0, 8}, Dims3{1, num_anchors, 1}, Dims3{1, 1, 1});
        auto yolov5_conf = slice_layer->getOutput(0);
        auto shuffle_layer = network->addShuffle(*yolov5_conf);
        shuffle_layer->setReshapeDimensions(Dims2{1, num_anchors});
        yolov5_conf = shuffle_layer->getOutput(0);
        auto topk_layer = network->addTopK(*yolov5_conf, TopKOperation::kMAX, TOPK_NUM, 1 << 1);
        auto topk_idx = topk_layer->getOutput(1);
        auto gather_layer = network->addGather(*yolov5_output, *topk_idx, 1);
        gather_layer->setNbElementWiseDims(1);
        auto yolov5_output_topk = gather_layer->getOutput(0);
        yolov5_output_topk->setName("output-topk");
        network->markOutput(*yolov5_output_topk);
        network->unmarkOutput(*yolov5_output);
    }
    network->getInput(0)->setName("input");
    auto config = builder->createBuilderConfig();
    if (builder->platformHasFastFp16())
    {
        std::cout << "[INFO]: platform support fp16, enable fp16" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else
    {
        std::cout << "[INFO]: platform do not support fp16, enable fp32" << std::endl;
    }
    size_t free, total;
    //    cuMemGetInfo(&free, &total);
    //    std::cout << "[INFO]: total gpu mem: " << (total >> 20) << "MB, free gpu mem: " << (free >> 20) << "MB" << std::endl;
    //    std::cout << "[INFO]: max workspace size will use all of free gpu mem" << std::endl;
    config->setMaxWorkspaceSize(1 << 30);
    runtime = createInferRuntime(glogger);
    TRT_ASSERT(runtime != nullptr);
    auto plan = builder->buildSerializedNetwork(*network, *config);
    TRT_ASSERT(plan != nullptr);
    TRT_ASSERT((engine = runtime->deserializeCudaEngine(plan->data(), plan->size())) != nullptr);
    delete plan;
    delete config;
    delete parser;
    delete network;
    delete builder;
}

void TRTModule::build_engine_from_cache(const std::string &cache_file)
{
    std::cout << "[INFO]: build engine from cache" << std::endl;
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    auto buffer = std::make_unique<char[]>(sz);
    ifs.read(buffer.get(), sz);
    runtime = createInferRuntime(glogger);
    TRT_ASSERT(runtime != nullptr);
    TRT_ASSERT((engine = runtime->deserializeCudaEngine(buffer.get(), sz)) != nullptr);
}

void TRTModule::cache_engine(const std::string &cache_file)
{
    auto engine_buffer = engine->serialize();
    TRT_ASSERT(engine_buffer != nullptr);
    std::ofstream ofs(cache_file, std::ios::binary);
    ofs.write(static_cast<const char *>(engine_buffer->data()), engine_buffer->size());
    delete engine_buffer;
}

void TRTModule::operator()(const cv::Mat &src, std::vector<bbox_t> &det)
{
    // pre-process [RGB uint8 -> CHW float]
    det.clear();
    cv::Mat x;
    cv::Mat preprocessedImage;
    x = src;
    x.convertTo(x, CV_32F, 1.0 / 255);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(x, preprocessedImage);

    // run model
    cudaMemcpyAsync(device_buffer[input_idx], preprocessedImage.data, input_sz * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(device_buffer, stream, nullptr);
    cudaMemcpyAsync(output_buffer, device_buffer[output_idx], output_sz * sizeof(float), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    
    int output_count = TOPK_NUM;
    int output_stride = static_cast<int>(output_sz / TOPK_NUM);
    const bool full_grid_output = output_sz % 21 == 0 && output_sz != TOPK_NUM * 21;
    if (full_grid_output)
    {
        output_stride = 21;
        output_count = static_cast<int>(output_sz / output_stride);
    }
    LOGM_S("[TRTdetect] output size: %lu, count: %d, stride: %d%s",
           static_cast<unsigned long>(output_sz),
           output_count,
           output_stride,
           full_grid_output ? ", full-grid" : "");
    TRT_ASSERT(output_stride == 21 || output_stride >= 22);

    std::vector<cv::Rect> boxes_nms;
    std::vector<float> scores_nms;
    std::vector<bbox_t> temp_bboxes;
    boxes_nms.reserve(output_count);
    scores_nms.reserve(output_count);
    temp_bboxes.reserve(output_count);

    int grid_stride = 8;
    int grid_x_center = 0;
    int grid_y_center = 0;
    for (int i = 0; i < output_count; i++)
    {
        auto *box_buffer = output_buffer + i * output_stride;
        if (!full_grid_output && box_buffer[8] < LOGIT_THRESH)
            break;
        if (full_grid_output && box_buffer[8] < KEEP_THRES)
        {
            grid_x_center += grid_stride;
            grid_x_center = (grid_x_center == src.cols) ? 0 : grid_x_center;
            grid_y_center += grid_x_center == 0 ? grid_stride : 0;
            grid_y_center = (grid_y_center == src.rows) ? 0 : grid_y_center;
            grid_stride *= grid_x_center == 0 && grid_y_center == 0 ? 2 : 1;
            continue;
        }

        bbox_t box;
        box.confidence = full_grid_output ? box_buffer[8] : sigmoid(box_buffer[8]);
        if (output_stride == 21) {
            box.tag_id = argmax(box_buffer + 9, 8);
            box.color_id = argmax(box_buffer + 17, 2);
        } else {
            int color_id = argmax(box_buffer + 9, 4);
            if (color_id == 2 || color_id == 3)
                continue;

            int class_id = argmax(box_buffer + 13, 9);
            if (class_id == 7 || class_id == 8) {
                class_id = 9;
            } else if (class_id == 0) {
                class_id = 7;
            } else if (class_id == 6) {
                class_id = 8;
            }

            if (color_id == 0) {
                color_id = 1;
            } else if (color_id == 1) {
                color_id = 0;
            }
            box.color_id = color_id;
            box.tag_id = class_id;
        }
        box.source = DetectionSource::NEURAL_NETWORK;

        float x_min = 1e5f;
        float y_min = 1e5f;
        float x_max = -1e5f;
        float y_max = -1e5f;

        for (int k = 0; k < 4; k++)
        {
            float x = box_buffer[2 * k];
            float y = box_buffer[2 * k + 1];
            if (full_grid_output)
            {
                x = x * 2.0f * grid_stride + grid_x_center;
                y = y * 2.0f * grid_stride + grid_y_center;
            }
            box.pts[k].x = x;
            box.pts[k].y = y;
            x_min = std::min(x_min, x);
            x_max = std::max(x_max, x);
            y_min = std::min(y_min, y);
            y_max = std::max(y_max, y);
        }

        temp_bboxes.push_back(box);

        float w = x_max - x_min;
        float h = y_max - y_min;
        boxes_nms.push_back(cv::Rect(x_min - 0.1f * w, y_min - 0.1f * h, w * 1.2f, h * 1.2f));
        scores_nms.push_back(box.confidence);

        if (full_grid_output)
        {
            grid_x_center += grid_stride;
            grid_x_center = (grid_x_center == src.cols) ? 0 : grid_x_center;
            grid_y_center += grid_x_center == 0 ? grid_stride : 0;
            grid_y_center = (grid_y_center == src.rows) ? 0 : grid_y_center;
            grid_stride *= grid_x_center == 0 && grid_y_center == 0 ? 2 : 1;
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_nms, scores_nms, CONF_THRESH, NMS_THRESH, indices);

    det.reserve(indices.size());
    for (int idx : indices) {
        det.push_back(temp_bboxes[idx]);
    }

    std::sort(det.begin(), det.end(), std::greater<bbox_t>());
}
