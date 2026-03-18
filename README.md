# YOLOv8 TensorRT C++ 推理引擎

基于 TensorRT C++ API 实现的 YOLOv8 目标检测推理引擎，包含完整的 CUDA 前处理、解码、NMS 以及双缓冲流水线优化。

---

## 项目简介

本项目在 [triple-Mu/YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) 的基础上，独立实现了以下优化：

- **CUDA letterbox 前处理 kernel**：将图像缩放、灰边填充、BGR→RGB、归一化、HWC→CHW 五步合一，替代 OpenCV CPU 实现
- **CUDA decode kernel**：并行解析 TensorRT 原始输出 `[1, 84, 8400]`，使用 `atomicAdd` 并行写入结果
- **CUDA NMS kernel**：替代 CPU 串行 NMS
- **双缓冲流水线**：GPU 推理与下一帧 CPU imread 并行执行
- **CUDA Graph**：将整帧推理流程录制为 Graph 一次提交，降低帧间延迟抖动
- **Pinned Memory**：使用页锁定内存加速 D2H 数据传输

---

## 性能对比

测试环境：NVIDIA MX450，CUDA 12.4，TensorRT 10.15.1，Windows 11

### 前处理耗时

| 版本 | 前处理耗时 | 说明 |
|------|-----------|------|
| OpenCV CPU letterbox | ~1.6ms | 原始版本 |
| CUDA kernel（含 malloc） | ~1.7ms | 无收益，malloc 开销抵消 |
| CUDA kernel（预分配显存） | ~0.5ms | **提升 3x** |

### 批量推理吞吐（测试6张图片）

| 版本 | 总耗时 | 每张均值 | 说明 |
|------|--------|---------|------|
| 串行原版 | ~105ms | ~17.5ms | baseline |
| 双缓冲 | ~75ms | ~12.5ms | 提升 29% |
| 双缓冲 + 异步 imread | ~65ms | ~10.8ms | 提升 38% |
| + CUDA Graph | ~65ms | ~10.8ms | 延迟抖动明显降低 |

> 注：MX450 上瓶颈为 cv::imread（~10ms），GPU 推理（~5ms）已被完全掩盖。CUDA Graph 主要收益为降低帧间抖动而非平均耗时。

### Nsight Systems Profiling

- letterbox CUDA kernel 占 GPU 总耗时约 **0.5%**，前处理不再是瓶颈
- 推理瓶颈为 TensorRT 内部 kernel（约 99.5%），已由 TensorRT 自动优化

---

## 环境要求

| 依赖 | 版本 |
|------|------|
| CUDA | 12.4 |
| TensorRT | 10.x（zip 包，含头文件） |
| OpenCV | 4.x |
| CMake | 3.12+ |
| MSVC | VS 2022 |

---

## 编译步骤

### 1. 导出模型

```bash
# 依赖项
# ultralytics  
# onxx  
# onnxsim  
# onxxslim  
# onnxruntime-gpu  
# tensorrt_cu12  
# tensorrt_cu12_bindings  
# tensorrt_cu12_libs  
# torch  
# opencv-python  
# nvidia-cudnn-cu12  
# nvidia-cublas-cu12  
# 等其他常见库不一一列出如报缺少自行安装

# End2End 模式（含 NMS，对应 main.cpp）
python export-det.py \
    --weights yolov8n.pt \
    --iou-thres 0.65 \
    --conf-thres 0.25 \
    --topk 100 \
    --opset 11 \
    --sim \
    --input-shape 1 3 640 640 \
    --device cuda:0

# 转换 TensorRT Engine
python build.py \
    --weights yolov8n.onnx \
    --iou-thres 0.65 \
    --conf-thres 0.25 \
    --topk 100 \
    --fp16 \
    --device cuda:0

# Normal 模式（不含 NMS，对应 main_normal.cpp）
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', opset=11, simplify=False)"

# 用 trtexec 转换
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_normal.engine --fp16
```

### 2. 编译

在`csrc/detect/end2end`中修改你的cuda和msvc编译器的位置

在 **x64 Native Tools Command Prompt for VS 2022** 中执行：

```bash
cd csrc/detect/end2end
mkdir build && cd build

cmake .. \
    -DTensorRT_ROOT="path/to/TensorRT" \
    -DOpenCV_DIR="path/to/opencv" \
    -DCMAKE_CUDA_ARCHITECTURES=your_architectures

cmake --build . --config Release
```

> **注意**：
> - `TensorRT_ROOT` 指向解压后的 TensorRT zip 目录（需包含 `include/` 和 `lib/`）
> - `CMAKE_CUDA_ARCHITECTURES` 根据你的 GPU 设置（MX450 为 75，RTX 30xx 为 86）
> - 需要将 TensorRT/bin 和 OpenCV/bin 加入 PATH 或复制到 exe 同目录

### 3. 运行

```bash
cd build/Release

# End2End 版本（baseline）
yolov8.exe path/to/yolov8n.engine path/to/image_or_dir

# 优化版本（CUDA 全流程 + CUDA Graph）
yolov8_normal.exe path/to/yolov8n_normal.engine path/to/image_or_dir
```

---

## 文件结构

```
csrc/detect/end2end/
├── CMakeLists.txt
├── main.cpp              # End2End 推理（baseline，含 NMS engine）
├── main_normal.cpp       # 完整优化版本（自实现 decode + NMS + CUDA Graph）
├── preprocess.cu         # CUDA letterbox 前处理 kernel
├── postprocess.cu        # CUDA decode + NMS kernel
├── cmake/
│   └── FindTensorRT.cmake
│   └── Function.cmake
└── include/
    ├── yolov8.hpp        # End2End 推理类
    ├── preprocess.h
    ├── postprocess.h
    └── common.hpp
```

---

## 主要优化说明

### CUDA Letterbox 前处理

将以下五步合并为单一 CUDA kernel，避免多次全局内存访问：

1. 等比例缩放（双线性插值）
2. 灰边填充（值 114/255）
3. BGR → RGB
4. uint8 → float32 归一化
5. HWC → CHW 格式转换

通过预分配显存（最大支持 4K 输入），消除每帧 `cudaMalloc/cudaFree` 开销，前处理耗时从 1.6ms 降至 0.5ms。

### 双缓冲流水线

使用两个独立的推理上下文，使 GPU 推理与下一帧的 CPU imread 并行执行：

```
帧i：  [imread] → [前处理] → [推理] → [后处理]
帧i+1：          [imread] → [前处理] → [推理] → [后处理]
```

### CUDA Graph

将单帧完整推理流程（前处理→推理→解码→NMS→D2H）录制为 CUDA Graph，每帧只需一次 `cudaGraphLaunch`，降低 CPU kernel launch overhead，帧间延迟更稳定。

动态参数（图像尺寸、缩放比例等）通过 GPU 上的参数结构体传递，无需重新录制 Graph。

### CUDA NMS + Bitonic Sort

实现 CUDA 并行 NMS，替代 CPU 串行实现：

- 使用 Bitonic Sort 对候选框按置信度降序排列
- 排序后 NMS 只需向前比较（`j < i`），理论减少约 50% IoU 计算次数
- 使用 `atomicAdd` 并行写入 decode 结果

**适用场景说明**：在密集检测场景（大量候选框）下收益明显。当前测试场景（有效框 27~45 个）中，框数量较少，整体耗时无明显差异。

---

## 参考

- [triple-Mu/YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- [NVIDIA TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
