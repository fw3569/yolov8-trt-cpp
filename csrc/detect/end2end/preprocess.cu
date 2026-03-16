#include "preprocess.h"

__global__ void letterbox_kernel(
    uint8_t* src,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    float scale,
    int pad_w,
    int pad_h,
    float fill_value)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height) return;

    float src_x = (dx - pad_w) / scale;
    float src_y = (dy - pad_h) / scale;

    float r, g, b;

    if (src_x < 0 || src_x >= src_width || src_y < 0 || src_y >= src_height) {
        r = g = b = fill_value;
    } else {
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = min(x0 + 1, src_width - 1);
        int y1 = min(y0 + 1, src_height - 1);

        float wx = src_x - x0;
        float wy = src_y - y0;
        float w00 = (1-wx) * (1-wy);
        float w01 = wx     * (1-wy);
        float w10 = (1-wx) * wy;
        float w11 = wx     * wy;

        uint8_t* p00 = src + (y0 * src_width + x0) * 3;
        uint8_t* p01 = src + (y0 * src_width + x1) * 3;
        uint8_t* p10 = src + (y1 * src_width + x0) * 3;
        uint8_t* p11 = src + (y1 * src_width + x1) * 3;

        b = (p00[0]*w00 + p01[0]*w01 +
             p10[0]*w10 + p11[0]*w11) / 255.f;
        g = (p00[1]*w00 + p01[1]*w01 +
             p10[1]*w10 + p11[1]*w11) / 255.f;
        r = (p00[2]*w00 + p01[2]*w01 +
             p10[2]*w10 + p11[2]*w11) / 255.f;
    }

    int area = dst_width * dst_height;
    dst[0 * area + dy * dst_width + dx] = r;
    dst[1 * area + dy * dst_width + dx] = g;
    dst[2 * area + dy * dst_width + dx] = b;
}

void preprocess_kernel_invoker(
    uint8_t* src,
    int src_width,
    int src_height,
    float* dst,
    int dst_width,
    int dst_height,
    float fill_value,
    cudaStream_t stream)
{
    float scale = std::min(
        (float)dst_width  / src_width,
        (float)dst_height / src_height
    );
    int pad_w = (dst_width  - src_width  * scale) / 2;
    int pad_h = (dst_height - src_height * scale) / 2;

    dim3 block(16, 16);
    dim3 grid(
        (dst_width  + block.x - 1) / block.x,
        (dst_height + block.y - 1) / block.y
    );

    letterbox_kernel<<<grid, block, 0, stream>>>(
        src, src_width, src_height,
        dst, dst_width, dst_height,
        scale, pad_w, pad_h, fill_value
    );
}

void preprocess(
    cv::Mat& image,
    uint8_t* src_device,
    float* dst_device,
    int dst_width,
    int dst_height,
    cudaStream_t stream)
{
    int src_size = image.rows * image.cols * 3;
    cudaMemcpyAsync(src_device, image.data, src_size,
                    cudaMemcpyHostToDevice, stream);

    preprocess_kernel_invoker(
        src_device,
        image.cols, image.rows,
        dst_device,
        dst_width, dst_height,
        114.f / 255.f,
        stream
    );
}