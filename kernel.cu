
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "EasyBMP.h"

texture<float, 1, cudaReadModeElementType> tex;

void writefile(float* image, int height, int width, bool gpu = false) {
    BMP output;
    output.SetSize(width, height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            RGBApixel pixel;
            pixel.Red = image[i * width + j];
            pixel.Green = image[i * width + j];
            pixel.Blue = image[i * width + j];
            output.SetPixel(j, i, pixel);
        }
    }
    output.WriteToFile(gpu ? "output_gpu.bmp" : "output_cpu.bmp");
}

float g(int x, int y, int x_0, int y_0, float sigma_d) {
    return exp(-((x - x_0) * (x - x_0) + (y - y_0) * (y - y_0)) / 2 / sigma_d / sigma_d);
}

float r(float i, float i_0, float sigma_r) {
    return exp(- (i - i_0) * (i - i_0) / 2 / sigma_r / sigma_r);
}

void bilateral_filter_cpu(float* output, float* input, int height, int width, float sigma_d, float sigma_r) {
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            float k = 0;
            float h = 0;
            for (size_t window_y = i - 1; window_y < i + 2; ++window_y) {
                for (size_t window_x = j - 1; window_x < j + 2; ++window_x) {
                    if (window_y >= 0 && window_y < height && window_x >= 0 && window_x < width) {
                        float w = g(window_y, window_x, i, j, sigma_d) * r(input[window_y * width + window_x], input[i * width + j], sigma_r);
                        k += w;
                        h += input[window_y * width + window_x] * w;
                    }
                }
            }
            output[i * width + j] = h / k;
        }
    }
}


__device__ float g_gpu(int x, int y, int x_0, int y_0, float sigma_d) {
    return __expf(-((x - x_0) * (x - x_0) + (y - y_0) * (y - y_0)) / 2. / sigma_d / sigma_d);
}

__device__ float r_gpu(float i, float i_0, float sigma_r) {
    return __expf(- (i - i_0) * (i - i_0) / 2. / sigma_r / sigma_r);
}


__global__ void bilateral_filter_gpu(float* output, int height, int width, float sigma_d, float sigma_r) {

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < height && col < width) {

        float window[9];

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                    window[(i + 1) * 3 + j + 1] = tex1Dfetch(tex, (row + i) * width + col +j);
                }
                else
                    window[(i + 1) * 3 + j + 1] = -1;
            }
        }

        
        float k = 0;
        float h = 0;

        for (size_t window_y = 0; window_y < 3; ++window_y) {
            for (size_t window_x = 0; window_x < 3; ++window_x) {
                float curr_i = window[window_y * 3 + window_x];
                if (curr_i != -1) {
                    float w = g_gpu(window_y, window_x, 1, 1, sigma_d) * r_gpu(curr_i, window[4], sigma_r);
                    k += w;
                    h += curr_i * w;
                }
            }
        }
        output[row * width + col] = h / k;
    }
}

int main()
{
    BMP Image;
    Image.ReadFromFile("input.bmp");
    int height = Image.TellHeight();
    int width = Image.TellWidth();

    float* image_array = new float[height * width];
    float* outputCPU = new float[height * width];
    float* outputGPU = new float[height * width];

    for (int j = 0; j < Image.TellHeight(); j++) {
        for (int i = 0; i < Image.TellWidth(); i++) {
            image_array[j * width + i] = Image(i, j)->Red;
        }
    }

    printf("Image size:%dx%d\n", height, width);

    float sigma_d, sigma_r;

    std::cout << "Enter sigma_d:\t";
    std::cin >> sigma_d;
    std::cout << "Enter sigma_r:\t";
    std::cin >> sigma_r;

    clock_t start, end;

    start = clock();
    bilateral_filter_cpu(outputCPU, image_array, height, width, sigma_d, sigma_r);
    end = clock();

    float cpu_time = static_cast <float>(end - start) / static_cast <float>(CLOCKS_PER_SEC);

    printf("CPU timme: %f sec.\n", cpu_time);

    writefile(outputCPU, height, width);

    dim3 block_dim(32, 32);
    dim3 grid_dim(ceil(static_cast <float> (width) / static_cast <float> (block_dim.x)), ceil(static_cast <float> (height) / static_cast <float> (block_dim.y)));

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    float* dev_img;
    cudaEventRecord(begin, 0);
    cudaMalloc(&dev_img, height * width * sizeof(float));
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
    cudaBindTexture(NULL, tex, dev_img, width * height * sizeof(float));
    cudaMemcpy(dev_img, image_array, height * width * sizeof(float), cudaMemcpyHostToDevice);
    float* dev_out;
    cudaMalloc(&dev_out, height * width * sizeof(float));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time_tex;
    cudaEventElapsedTime(&gpu_time_tex, begin, stop);

    printf("Malloc, memcpy, bind texture: %f sec.\n", gpu_time_tex/1000.);

    cudaEventRecord(begin, 0);
    bilateral_filter_gpu << <grid_dim, block_dim >> > (dev_out, height, width, sigma_d, sigma_r);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, begin, stop);

    printf("Filtering: %f sec.\n", gpu_time/1000.);

    cudaEventRecord(begin, 0);
    cudaMemcpy(outputGPU, dev_out, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time_memcpy;
    cudaEventElapsedTime(&gpu_time_memcpy, begin, stop);

    printf("Memcpy: %f sec.\n", gpu_time_memcpy/1000.);

    float total = (gpu_time + gpu_time_memcpy + gpu_time_tex) / 1000.;

    writefile(outputGPU, height, width, true);

    printf("Total GPU time: %f sec.\nSpeedup with total time: %f\nSpeedup with only calc: %f\n", total, cpu_time / total, cpu_time / (gpu_time / 1000.));

    cudaFree(dev_img);
    cudaFree(dev_out);

    delete[] image_array;
    delete[] outputCPU;
    delete[] outputGPU;

    return 0;
}