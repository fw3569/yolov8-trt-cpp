#pragma once
#include <cuda_runtime.h>
#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float score;
    int   label;
};

void decode_kernel_invoker(
    float*      output,
    int         num_anchors,
    int         num_classes,
    float       conf_thres,
    float*      boxes_device,
    float*      scores_device,
    float*      scores_sort_device,
    int*        indices_device,
    int*        labels_device,
    int*        num_valid,
    cudaStream_t stream
);

void nms_kernel_invoker(
    float*      boxes_device,
    int*        indices_device,
    int*        labels_device,
    int         num_anchors,
    int*        num_valid,
    float       iou_thres,
    int*        keep_device,
    cudaStream_t stream
);

void bitonic_sort_invoker(
    float*       scores,
    int*         indices,
    int*         num_valid,
    int          n,
    cudaStream_t tream);
