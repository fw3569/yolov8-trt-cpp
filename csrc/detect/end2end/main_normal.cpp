#include "opencv2/opencv.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include <NvInfer.h>
#include <fstream>
#include <chrono>
#include <future>

#include "common.hpp"

namespace fs = ghc::filesystem;

const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct InferContext {
    nvinfer1::IRuntime*          runtime  = nullptr;
    nvinfer1::ICudaEngine*       engine   = nullptr;
    nvinfer1::IExecutionContext* context  = nullptr;
    cudaStream_t                 stream   = nullptr;
    cudaEvent_t                  decode_finished_event = nullptr;
    Logger                       logger{nvinfer1::ILogger::Severity::kERROR};

    void* d_input   = nullptr;  // [1,3,640,640]
    void* d_output  = nullptr;  // [1,84,8400]
    uint8_t* d_src  = nullptr;
    int src_size    = 0;

    float* d_boxes       = nullptr;
    float* d_scores      = nullptr;
    float* d_scores_sort = nullptr;
    int*   d_indices     = nullptr;
    int*   d_labels      = nullptr;
    int*   d_num_valid   = nullptr;
    int*   d_keep        = nullptr;
    
    float* h_boxes     = nullptr;
    float* h_scores    = nullptr;
    int*   h_labels    = nullptr;
    int*   h_num_valid = nullptr;
    int*   h_keep      = nullptr;

    float ratio, dw, dh, width, height;

    cudaGraph_t     graph      = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool            graph_ready = false;
    PreprocessParams* d_params = nullptr;
    PreprocessParams  h_params;
};

InferContext* create_context(const std::string& engine_path) {
    auto* ctx = new InferContext();
    
    std::ifstream file(engine_path, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);

    ctx->runtime = nvinfer1::createInferRuntime(ctx->logger);
    ctx->engine  = ctx->runtime->deserializeCudaEngine(buf.data(), size);
    ctx->context = ctx->engine->createExecutionContext();
    cudaStreamCreate(&ctx->stream);
    cudaEventCreate(&ctx->decode_finished_event);

    cudaMalloc(&ctx->d_input,  1*3*640*640*sizeof(float));
    cudaMalloc(&ctx->d_output, 1*84*8400*sizeof(float));
    cudaMalloc(&ctx->d_src,    3840*2160*3);
    ctx->src_size = 3840*2160*3;

    cudaMalloc(&ctx->d_boxes,       8400*4*sizeof(float));
    cudaMalloc(&ctx->d_scores,      8400*sizeof(float));
    cudaMalloc(&ctx->d_scores_sort, 16384*sizeof(float)); // pad to 2^14
    cudaMalloc(&ctx->d_indices,     16384*sizeof(int));   // pad to 2^14
    cudaMalloc(&ctx->d_labels,      8400*sizeof(int));
    cudaMalloc(&ctx->d_num_valid,   sizeof(int));
    cudaMalloc(&ctx->d_keep,        8400*sizeof(int));
    cudaMalloc(&ctx->d_params,      sizeof(PreprocessParams));

    cudaMallocHost(&ctx->h_boxes,     8400*4*sizeof(float));
    cudaMallocHost(&ctx->h_scores,    8400*sizeof(float));
    cudaMallocHost(&ctx->h_labels,    8400*sizeof(int));
    cudaMallocHost(&ctx->h_num_valid, sizeof(int));
    cudaMallocHost(&ctx->h_keep,      8400*sizeof(int));

    ctx->context->setInputShape("images",
        nvinfer1::Dims{4, {1, 3, 640, 640}});
    ctx->context->setTensorAddress("images",  ctx->d_input);
    ctx->context->setTensorAddress("output0", ctx->d_output);

    return ctx;
}

void destroy_context(InferContext* ctx) {
    if (ctx->graph_exec) cudaGraphExecDestroy(ctx->graph_exec);
    cudaStreamSynchronize(ctx->stream);
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_output);
    cudaFree(ctx->d_src);
    cudaFree(ctx->d_boxes);
    cudaFree(ctx->d_scores);
    cudaFree(ctx->d_scores_sort);
    cudaFree(ctx->d_indices);
    cudaFree(ctx->d_labels);
    cudaFree(ctx->d_num_valid);
    cudaFree(ctx->d_keep);
    cudaFree(ctx->d_params);
    cudaFreeHost(ctx->h_boxes);
    cudaFreeHost(ctx->h_scores);
    cudaFreeHost(ctx->h_labels);
    cudaFreeHost(ctx->h_num_valid);
    cudaFreeHost(ctx->h_keep);

    cudaEventDestroy(ctx->decode_finished_event);
    cudaStreamDestroy(ctx->stream);
    delete ctx->context;
    delete ctx->engine;
    delete ctx->runtime;

    delete ctx;
}

void infer(InferContext* cur_ctx, InferContext* prev_ctx,
           const cv::Mat& image, std::vector<Detection>& results)
{
    if (cur_ctx != nullptr) {
        InferContext* ctx = cur_ctx;
        int W = 640, H = 640;

        int src_size = image.rows * image.cols * 3;
        if (src_size > ctx->src_size) {
            cudaFree(ctx->d_src);
            cudaMalloc(&ctx->d_src, src_size);
            ctx->src_size = src_size;
            ctx->graph_ready = false;
        }
        cudaMemcpyAsync(ctx->d_src, image.data, src_size,
                        cudaMemcpyHostToDevice, ctx->stream);

        ctx->h_params = compute_params(image.cols, image.rows, W, H);
        cudaMemcpyAsync(ctx->d_params, &ctx->h_params,
                        sizeof(PreprocessParams),
                        cudaMemcpyHostToDevice, ctx->stream);

        ctx->dw    = ctx->h_params.pad_w;
        ctx->dh    = ctx->h_params.pad_h;
        ctx->ratio = 1.f / ctx->h_params.scale;
        ctx->width  = image.cols;
        ctx->height = image.rows;

        cudaStreamSynchronize(ctx->stream);

        if (!ctx->graph_ready) {
            cudaStreamBeginCapture(ctx->stream,
                                   cudaStreamCaptureModeGlobal);

            preprocess_kernel_invoker(
                ctx->d_src,
                (float*)ctx->d_input,
                ctx->d_params,
                ctx->stream);

            ctx->context->enqueueV3(ctx->stream);

            cudaMemsetAsync(ctx->d_scores_sort, 0, 16384 * sizeof(float),
                            ctx->stream);
            decode_kernel_invoker((float*)ctx->d_output, 8400, 80, 0.25f,
                                  ctx->d_boxes, ctx->d_scores,
                                  ctx->d_scores_sort, ctx->d_indices,
                                  ctx->d_labels, ctx->d_num_valid, ctx->stream);

            cudaMemcpyAsync(ctx->h_num_valid, ctx->d_num_valid,
                            sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);
            cudaMemcpyAsync(ctx->h_boxes,  ctx->d_boxes,
                            8400*4*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
            cudaMemcpyAsync(ctx->h_scores, ctx->d_scores,
                            8400*sizeof(float), cudaMemcpyDeviceToHost, ctx->stream);
            cudaMemcpyAsync(ctx->h_labels, ctx->d_labels,
                            8400*sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);

            bitonic_sort_invoker(ctx->d_scores_sort, ctx->d_indices,
                                 ctx->d_num_valid, 16384, ctx->stream);
            nms_kernel_invoker(ctx->d_boxes, ctx->d_indices,
                               ctx->d_labels, 8400, ctx->d_num_valid, 0.65f,
                               ctx->d_keep, ctx->stream);

            cudaMemcpyAsync(ctx->h_keep,   ctx->d_keep,
                            8400*sizeof(int), cudaMemcpyDeviceToHost, ctx->stream);

            cudaGraph_t graph;
            cudaStreamEndCapture(ctx->stream, &graph);

            if (ctx->graph_exec) cudaGraphExecDestroy(ctx->graph_exec);
            cudaGraphInstantiate(&ctx->graph_exec, graph, nullptr, nullptr, 0);
            cudaGraphDestroy(graph);
            ctx->graph_ready = true;
        }

        cudaGraphLaunch(ctx->graph_exec, ctx->stream);
    }

    if (prev_ctx != nullptr) {
        InferContext* ctx = prev_ctx;
        cudaStreamSynchronize(ctx->stream);

        results.clear();
        for (int i = 0; i < *ctx->h_num_valid; i++) {
            if (!ctx->h_keep[i]) continue;
            Detection det;
            det.x1    = (ctx->h_boxes[i*4+0] - ctx->dw) * ctx->ratio;
            det.y1    = (ctx->h_boxes[i*4+1] - ctx->dh) * ctx->ratio;
            det.x2    = (ctx->h_boxes[i*4+2] - ctx->dw) * ctx->ratio;
            det.y2    = (ctx->h_boxes[i*4+3] - ctx->dh) * ctx->ratio;
            det.score = ctx->h_scores[i];
            det.label = ctx->h_labels[i];
            results.push_back(det);
        }
    }
}

void draw(cv::Mat& image, const std::vector<Detection>& dets) {
    for (auto& d : dets) {
        cv::rectangle(image,
            cv::Point((int)d.x1, (int)d.y1),
            cv::Point((int)d.x2, (int)d.y2),
            cv::Scalar(0, 255, 0), 2);
        char text[64];
        sprintf(text, "%s %.1f%%",
            CLASS_NAMES[d.label].c_str(), d.score * 100);
        cv::putText(image, text,
            cv::Point((int)d.x1, (int)d.y1 - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 255, 0), 1);
    }
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine] [image]\n", argv[0]);
        return -1;
    }
    cudaSetDevice(0);

    InferContext* ctx[2];
    ctx[0] = create_context(argv[1]);
    ctx[1] = create_context(argv[1]);

    const fs::path path{argv[2]};
    std::vector<std::string> imagePathList;

    if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    } else {
        imagePathList.push_back(path.string());
    }

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    // warmup
    constexpr int warmup_times = 10;
    std::vector<Detection> dets[2];
    for (int i=0;i<warmup_times;++i){
      cv::Mat image(640, 640, CV_8UC3);
      infer(ctx[i % 2], ctx[(i + 1) % 2], image, dets[i % 2]);
    }
    cudaStreamSynchronize(ctx[0]->stream);
    cudaStreamSynchronize(ctx[1]->stream);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    int n=imagePathList.size();
    cv::Mat image[2];
    std::future<cv::Mat> next_image_future;
    if (n > 0) {
        next_image_future = std::async(std::launch::async, [&]() {
            return cv::imread(imagePathList[0]);
        });
    }
    for (int i=0; i <= n; ++i) {
        if (i < n) {
            image[i % 2] = next_image_future.get();
        }
        if (i + 1 < n) {
            next_image_future = std::async(std::launch::async, [&imagePathList, i]() {
                return cv::imread(imagePathList[i + 1]);
            });
        }
        infer(i < n ? ctx[i % 2] : nullptr, i > 0 ? ctx[(i + 1) % 2] : nullptr, image[i % 2], dets[i % 2]);
        if(i > 0){
          draw(image[(i + 1) % 2], dets[i % 2]);
          // please delete these two lines when test time cost
          cv::imshow("result", image[(i + 1) % 2]);
          cv::waitKey(0);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
    printf("cost %.2f ms\n", ms);
    cv::destroyAllWindows();
    destroy_context(ctx[1]);
    destroy_context(ctx[0]);
    return 0;
}
