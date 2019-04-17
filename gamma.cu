#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cuda_device_runtime_api.h"

using namespace std;
using namespace cv;
using namespace cuda;

__global__ void gamma_kernel(PtrStepSz<uchar3> src, PtrStepSz<uchar3> dst, double *ptr) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if (x < src.cols && y < src.rows) {
		dst(y, x).x = ptr[src(y, x).x];
		dst(y, x).y = ptr[src(y, x).y];
		dst(y, x).z = ptr[src(y, x).z];
	}
	__syncthreads();
}

void gammaCorrect_Hawkeye(GpuMat src, GpuMat dst, double c, double gamma) {
	if (src.type() == CV_8UC1)
		cuda::cvtColor(src, src, CV_GRAY2BGR);

	const int rows = src.rows, cols = src.cols;
	double pixels[256] = { 0 }, *ptr = pixels;

	for (int i = 0; i < 256; i++) {
		*(pixels + i) = c * pow(((double)i / (double)255.0), gamma) * 255;
		*(pixels + i) = *(pixels + i) <= 255 ? *(pixels + i) : 255;
		*(pixels + i) = *(pixels + i) >= 0 ? *(pixels + i) : 0;
	}
	double *gpu_ptr;
	
	cudaMalloc((void**)&gpu_ptr, sizeof(double) * 256);
	cudaMemcpy(gpu_ptr, ptr, sizeof(double) * 256, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(32, 32);
	uint block_num_x = (cols + threadsPerBlock.x - 1) / threadsPerBlock.x;
	uint block_num_y = (rows + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(block_num_x, block_num_y);

	gamma_kernel << <numBlocks, threadsPerBlock >> > (src, dst, gpu_ptr);
}
int main()
{
	int num_devices = getCudaEnabledDeviceCount();
	if (num_devices <= 0) {
		cerr << "No device detected!" << endl;
		return -1;
	}

	int enable_device_id = -1;
	for (int i = 0; i < num_devices; i++) {
		DeviceInfo deviceInfo(i);
		if (deviceInfo.isCompatible())
			enable_device_id = i;
	}
	if (enable_device_id < 0) {
		cerr << "GPU module incompatible!" << endl;
		return -1;
	}

	setDevice(enable_device_id);

	Mat src = imread("F:/Python Projects/Dong/Dump/dump5.png"), dst;
	GpuMat gpu_src, gpu_dst;

	gpu_src.upload(src);

	if (!gpu_src.data)
		return -1;

	gpu_dst = GpuMat(gpu_src.rows, gpu_src.cols, CV_8UC3, Scalar(0, 0, 0));

	gammaCorrect_Hawkeye(gpu_src, gpu_dst, 1, 0.8);

	gpu_dst.download(dst);
	
	return 0;
}