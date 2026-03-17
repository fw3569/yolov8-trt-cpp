#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct PreprocessParams {
    int   src_width;
    int   src_height;
    int   dst_width;
    int   dst_height;
    float scale;
    int   pad_w;
    int   pad_h;
    float fill_value;
};

PreprocessParams compute_params(
    int src_w, int src_h,
    int dst_w, int dst_h,
    float fill_value = 114.f/255.f
);

void preprocess_kernel_invoker(
    uint8_t*          src,
    float*            dst,
    PreprocessParams* params,
    cudaStream_t      stream
);
