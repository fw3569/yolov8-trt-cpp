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

__global__ void bitonic_sort_step(
    float* scores,
    int*   indices,
    int*   num_valid,
    int    stage,
    int    step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int j = i ^ step;
    if (j > i) {
        bool ascending = (i & stage) == 0;
        if ((scores[i] < scores[j]) == ascending) {
            float tmp_s  = scores[i];
            scores[i]    = scores[j];
            scores[j]    = tmp_s;
            int tmp_idx = indices[i];
            indices[i]  = indices[j];
            indices[j]  = tmp_idx;
        }
    }
}

void bitonic_sort_invoker(
    float* scores,
    int* indices,
    int* num_valid,
    int n,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;

    for (int stage = 2; stage <= n; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            bitonic_sort_step<<<grid, block, 0, stream>>>(
                scores, indices, num_valid, stage, step);
        }
    }
}

__global__ void init_keep(int* keep, int* num_valid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < *num_valid) keep[i] = 1;
}

__global__ void nms_kernel(
    float* boxes,
    int*   indices,
    int*   labels,
    int*   num_valid,
    float  iou_thres,
    int*   keep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *num_valid) return;

    int id_i = indices[i];

    for (int j = 0; j < i; j++) {
        int id_j = indices[j];
        if (!keep[id_j]) continue;
        if (labels[id_j] != labels[id_i]) continue;

        if (iou(boxes + id_i * 4, boxes + id_j * 4) > iou_thres) {
            keep[id_i] = 0;
            return;
        }
    }
}

void nms_kernel_invoker(
    float* boxes_device,
    int*   indices_device,
    int*   labels_device,
    int    num_anchors,
    int*   num_valid,
    float  iou_thres,
    int*   keep_device,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (num_anchors + block - 1) / block;
    init_keep<<<grid, block, 0, stream>>>(keep_device, num_valid);
    nms_kernel<<<grid, block, 0, stream>>>(
        boxes_device, indices_device,
        labels_device, num_valid, iou_thres, keep_device);
}

__global__ void decode_kernel(
    float* output,
    int    num_anchors,
    int    num_classes,
    float  conf_thres,
    float* boxes_device,
    float* scores_device,
    float* scores_sort_device,
    int*   indices_device,
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
    scores_sort_device[pos]    = max_score;
    indices_device[pos]        = pos;
    labels_device[pos]         = max_label;
}

void decode_kernel_invoker(
    float* output,
    int    num_anchors,
    int    num_classes,
    float  conf_thres,
    float* boxes_device,
    float* scores_device,
    float* scores_sort_device,
    int*   indices_device,
    int*   labels_device,
    int*   num_valid,
    cudaStream_t stream)
{
    cudaMemsetAsync(num_valid, 0, sizeof(int), stream);

    int block = 256;
    int grid  = (num_anchors + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(
        output, num_anchors, num_classes, conf_thres, boxes_device,
        scores_device, scores_sort_device, indices_device,
        labels_device, num_valid);
}
