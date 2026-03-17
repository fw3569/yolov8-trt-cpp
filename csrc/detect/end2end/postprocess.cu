#include "postprocess.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>

__device__ float iou(float* a, float* b)
{
    float x1 = fmaxf(a[0], b[0]);
    float y1 = fmaxf(a[1], b[1]);
    float x2 = fminf(a[2], b[2]);
    float y2 = fminf(a[3], b[3]);

    float inter = fmaxf(0.f, x2 - x1) * fmaxf(0.f, y2 - y1);
    float area_a = (a[2]-a[0]) * (a[3]-a[1]);
    float area_b = (b[2]-b[0]) * (b[3]-b[1]);
    return inter / (area_a + area_b - inter);
}

__global__ void init_keep(int* keep, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keep[i] = 1;
}

__global__ void nms_kernel(
    float* boxes,
    float* scores,
    int*   labels,
    int*   num_boxes,
    float  iou_thres,
    int*   keep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *num_boxes) return;

    keep[i] = 1;
    __syncthreads();

    for (int j = 0; j < *num_boxes; j++) {
        if (j == i) continue;
        if (labels[j] != labels[i]) continue;
        if (scores[j] <= scores[i]) continue;

        if (iou(boxes + i * 4, boxes + j * 4) > iou_thres) {
            keep[i] = 0;
            return;
        }
    }
}

void nms_kernel_invoker(
    float* boxes_device,
    float* scores_device,
    int*   labels_device,
    int    num_anchors,
    int*   num_boxes,
    float  iou_thres,
    int*   keep_device,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (num_anchors + block - 1) / block;
    init_keep<<<grid, block, 0, stream>>>(
        keep_device, num_anchors);
    nms_kernel<<<grid, block, 0, stream>>>(
        boxes_device, scores_device, labels_device,
        num_boxes, iou_thres, keep_device);
}

__global__ void decode_kernel(
    float* output,
    int    num_anchors,
    int    num_classes,
    float  conf_thres,
    float* boxes_device,
    float* scores_device,
    int*   labels_device,
    int*   num_valid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    float cx = output[0 * num_anchors + idx];
    float cy = output[1 * num_anchors + idx];
    float w  = output[2 * num_anchors + idx];
    float h  = output[3 * num_anchors + idx];

    float max_score = 0.f;
    int   max_label = 0;
    for (int c = 0; c < num_classes; c++) {
        float score = output[(4 + c) * num_anchors + idx];
        if (score > max_score) {
            max_score = score;
            max_label = c;
        }
    }

    if (max_score < conf_thres) return;

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

    int pos = atomicAdd(num_valid, 1);
    boxes_device[pos * 4 + 0] = x1;
    boxes_device[pos * 4 + 1] = y1;
    boxes_device[pos * 4 + 2] = x2;
    boxes_device[pos * 4 + 3] = y2;
    scores_device[pos]         = max_score;
    labels_device[pos]         = max_label;
}

void decode_kernel_invoker(
    float* output,
    int    num_anchors,
    int    num_classes,
    float  conf_thres,
    float* boxes_device,
    float* scores_device,
    int*   labels_device,
    int*   num_valid,
    cudaStream_t stream)
{
    cudaMemsetAsync(num_valid, 0, sizeof(int), stream);

    int block = 256;
    int grid  = (num_anchors + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(
        output, num_anchors, num_classes, conf_thres,
        boxes_device, scores_device, labels_device, num_valid);
}
