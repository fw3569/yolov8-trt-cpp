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
    int*        labels_device,
    int*        num_valid,
    cudaStream_t stream
);

void nms_kernel_invoker(
    float*      boxes_device,
    float*      scores_device,
    int*        labels_device,
    int         num_anchors,
    int*        num_boxes,
    float       iou_thres,
    int*        keep_device,
    cudaStream_t stream
);
