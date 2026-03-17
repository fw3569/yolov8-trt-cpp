#include "preprocess.h"

__global__ void letterbox_kernel(
    uint8_t*          src,
    float*            dst,
    PreprocessParams* params)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    int dst_w = params->dst_width;
    int dst_h = params->dst_height;
    if (dx >= dst_w || dy >= dst_h) return;

    float src_x = (dx - params->pad_w) / params->scale;
    float src_y = (dy - params->pad_h) / params->scale;

    float r, g, b;
    int src_w = params->src_width;
    int src_h = params->src_height;

    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        r = g = b = params->fill_value;
    } else {
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);

        float wx = src_x - x0;
        float wy = src_y - y0;

        float w00 = (1-wx) * (1-wy);
        float w01 = wx     * (1-wy);
        float w10 = (1-wx) * wy;
        float w11 = wx     * wy;

        uint8_t* p00 = src + (y0 * src_w + x0) * 3;
        uint8_t* p01 = src + (y0 * src_w + x1) * 3;
        uint8_t* p10 = src + (y1 * src_w + x0) * 3;
        uint8_t* p11 = src + (y1 * src_w + x1) * 3;

        b = (p00[0]*w00 + p01[0]*w01 + p10[0]*w10 + p11[0]*w11) / 255.f;
        g = (p00[1]*w00 + p01[1]*w01 + p10[1]*w10 + p11[1]*w11) / 255.f;
        r = (p00[2]*w00 + p01[2]*w01 + p10[2]*w10 + p11[2]*w11) / 255.f;
    }

    int area = dst_w * dst_h;
    dst[0 * area + dy * dst_w + dx] = r;
    dst[1 * area + dy * dst_w + dx] = g;
    dst[2 * area + dy * dst_w + dx] = b;
}

PreprocessParams compute_params(
    int src_w, int src_h,
    int dst_w, int dst_h,
    float fill_value)
{
    PreprocessParams p;
    p.src_width  = src_w;
    p.src_height = src_h;
    p.dst_width  = dst_w;
    p.dst_height = dst_h;
    p.scale      = std::min((float)dst_w/src_w, (float)dst_h/src_h);
    p.pad_w      = (dst_w - src_w * p.scale) / 2;
    p.pad_h      = (dst_h - src_h * p.scale) / 2;
    p.fill_value = fill_value;
    return p;
}

void preprocess_kernel_invoker(
    uint8_t*          src,
    float*            dst,
    PreprocessParams* params,
    cudaStream_t      stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (640 + block.x - 1) / block.x,
        (640 + block.y - 1) / block.y
    );
    letterbox_kernel<<<grid, block, 0, stream>>>(src, dst, params);
}
