#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void preprocess_kernel_invoker(
    uint8_t* src,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    float fill_value,
    cudaStream_t stream
);

void preprocess(
    cv::Mat& image,
    uint8_t* src_device,
    float* dst_device,
    int dst_width,
    int dst_height,
    cudaStream_t stream
);