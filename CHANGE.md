# SHtech Auto Aim Local Build Changes

本文记录为在 NUC11 / Ubuntu 22.04 / CUDA 11.8 / TensorRT 8.6.1 / HikRobot MVS amd64 环境下完成编译、启动和 TensorRT 识别解码所做的源码修改。

## 目标状态

- `build-hik` 可成功编译 `auto-aim`。
- TensorRT 后端可链接 x86_64 平台的 CUDA / TensorRT 库。
- HikRobot MVS 64 位 SDK 可参与链接。
- `SKD250526.onnx` 的 TensorRT engine 可重新生成 cache。
- `stride=21` 的旧模型输出按完整 grid 解码，避免检测框坐标错误导致框不可见、PnP 距离异常和角点精修 ROI 越界。

## 编译与链接

### `detect/CMakeLists.txt`

- 增加 x86_64 TensorRT 库搜索路径：

```cmake
"/usr/lib/x86_64-linux-gnu/libnv*.so"
```

原因：原工程主要面向 aarch64，NUC11 是 x86_64，未加入该路径时 TensorRT 相关库不能被 CMake 正确收集和链接。

## TensorRT 后端

### `detect/TensorRT/TRTModule.hpp`

- 增加 `nvinfer1::IRuntime *runtime` 成员。

原因：原逻辑在反序列化 engine 后过早销毁 runtime，会触发 TensorRT 报错：

```text
Destroying a runtime before destroying deserialized engines created by the runtime leads to undefined behavior.
```

### `detect/TensorRT/TRTModule.cpp`

- 构造函数初始化 `runtime/context/engine`。
- 析构函数按 `context -> engine -> runtime` 顺序释放 TensorRT 对象。
- 将已弃用的 `enqueue(...)` 改为 `enqueueV2(...)`。
- 构建 engine 时不再假设固定 `6720` anchor 数，而是读取 ONNX 输出维度。
- 兼容两类模型输出：
  - `stride == 21`：旧模型格式。
  - `stride >= 22`：新模型格式，保留 color/class 映射逻辑。
- 修正 `stride=21` 的 TRT 解码路径：
  - 不再在 engine 内对旧模型做 topk gather。
  - 直接输出完整 grid。
  - 后处理按 AXCL 中已有逻辑解码：

```cpp
x = x * 2.0f * grid_stride + grid_x_center;
y = y * 2.0f * grid_stride + grid_y_center;
```

原因：旧模型前 8 个点不是绝对像素坐标，也不是简单 0-1 归一化坐标，而是 grid-relative 坐标。原 topk gather 丢失 grid index 后，无法恢复正确框位置，表现为：

```text
center=(0.x, 0.x)
dist=25m~50m
corner_refine FailStage roi=全部失败
```

- 旧 cache 已备份并移除，触发新 engine 生成：

```text
asset/models/SKD250526.cache.old-topk-griddecode-*
```

新 cache 应显示为完整 grid 输出：

```text
[TRTdetect] output size: 141120, count: 6720, stride: 21, full-grid
```

## 检测与可视化

### `detect/detect_submodule.cpp`

- 保留检测结果从模型输入坐标映射回原图坐标的逻辑。
- 将检测框显示从白色 1px 改为绿色 2px。
- 在四个角点绘制绿色圆点，便于确认框是否真的画到了图像上。

原因：之前框线过细且在坐标错误时挤在左上角，不利于区分“没有画框”和“坐标错误”。

### `detect/corner_refine_submodule.cpp`

- 将角点精修统计从仅文件日志扩展为终端日志。
- 增加失败阶段统计：

```text
unknown / input / roi / no_light / select / final_check
```

- 可视化框线同样改为绿色 2px 并绘制角点圆点。

原因：`predictor` 初始化新目标只接受 `DetectionSource::TRADITIONAL`，即角点精修成功的目标。终端必须能看到精修失败原因，否则会表现为检测到了但一直 `empty detection`。

### `detect/armor_corner_optimizer.hpp`

- 在 `CornerRefineCallStats` 中增加 `fail_stage` 字段。

### `detect/armor_corner_optimizer.cpp`

- 在以下失败路径记录 `fail_stage`：
  - 输入图为空。
  - ROI 越界或非法。
  - 未找到灯条。
  - 灯条选择失败。
  - 最终几何检查失败。
- 当左右灯条没有都成功选择时立即返回失败。

原因：区分角点精修失败到底来自 ROI、阈值、灯条选择还是最终几何约束。

## Predictor 过滤逻辑与诊断

### `predict/MultiPolicyPredictor_submodule.cpp`

- 在读取 PnP `measurement` 前先检查 `pnp_get_measurement(...)` 返回值。

原因：原代码在 PnP 失败时仍会读取未定义的 `measurement`。

- 修正几何过滤中的 yaw 维度：

```cpp
abs(measurement(3, 0)) < max_yaw_accept
```

原代码误用了 `measurement(2, 0)` 与 `max_yaw_accept` 比较，而 `CoordTransformer` 中 `measurement` 定义为：

```text
[y, x, z, absolute_yaw]
```

因此第 4 维 `measurement(3)` 才是 yaw。

- 增加 predictor 过滤汇总日志：

```text
[predict] filter summary:
detected / new / tracking / color_reject / pnp_fail / geometry_reject / source_reject / enemy_is_blue
```

- 增加几何过滤明细日志：

```text
[predict] geometry reject detail:
tag / color / source / center / dist / yaw / height
```

原因：定位 `empty detection` 到底来自颜色、PnP、几何阈值还是角点来源过滤。

## 运行时注意

- 如果启动时仍显示：

```text
[INFO]: build engine from cache
[TRTdetect] output size: 2688, topk: 128, stride: 21
```

说明仍在使用旧 topk cache，需要移除旧 `asset/models/SKD250526.cache` 并重新启动。

- 修正后的旧模型 TRT cache 应显示：

```text
[INFO]: build engine from onnx
[TRTdetect] output size: 141120, count: 6720, stride: 21, full-grid
```

- `predictor_show_image` 当前代码中基本没有绘制逻辑。检测框应在以下窗口观察：
  - `detect_submodule`
  - `corner_refine_submodule`

## 非源码环境处理

- TensorRT 最终使用 `8.6.1.6-1+cuda11.8`。
- HikRobot MVS 使用 amd64 版本，运行时应保留：

```bash
LD_LIBRARY_PATH=/opt/MVS/lib/64:...
```

- 已清理 32 位 MVS 路径 `/opt/MVS/lib/32`，避免运行时加载错误架构库。

