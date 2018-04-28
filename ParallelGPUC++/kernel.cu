#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include "cuComplex.h"

__device__ 
void PaintTheBelongingPoint(uint8_t* &colors, int &width, int &x, int &y)
{
	for (int j = 0; j < 3; j++)
		colors[y * width * 4 + x * 4 + j] = 0;
	colors[y * width * 4 + x * 4 + 3] = 255;
}

__device__ 
void PaintTheNotBelongingPoint(uint8_t* &colors, int &width, int &x, int &y, int &iter, int &calcIterNum, float &invCalcIterNum)
{
	colors[y * width * 4 + x * 4 + 0] =
		(255 - (calcIterNum - iter) * invCalcIterNum * 255.0f);
	colors[y * width * 4 + x * 4 + 2] =
		(255 - (calcIterNum - 0.5f * iter) * invCalcIterNum * 255.0f);
}

__global__ 
void FractalCalcOnDevice(uint8_t* colors, int* width, int* height, float* scale, float* offsetX, float* offsetY, int* calcIterNum)
{
	int _width = *width;
	int _height = *height;
	float _scale = *scale;
	float _offsetX = *offsetX;
	float _offsetY = *offsetY;
	int _calcIterNum = *calcIterNum;
	float invCalcIterNum = 1.0f / (_calcIterNum - 1);
	
	cuFloatComplex c = make_cuComplex(0.0f, 0.0f);
	cuFloatComplex z = make_cuComplex(0.0f, 0.0f);
	cuFloatComplex zPrev = make_cuComplex(cuCrealf(c), cuCimagf(c));

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float mathX = 0.0;
	float mathY = 0.0;

	int y = index / _width;
	int x = index % _width;
	
	while (y * _width + x < _width * _height)
	{
		mathX = (x - _width / 2 + _offsetX) / _scale;
		mathY = (_height / 2 - y - _offsetY) / _scale;

		//
		// Begin with the 1-st iteration. (Not with 0-th)
		c.x = mathX;
		c.y = mathY;
		z.x = 0.0f;
		z.y = 0.0f;
		zPrev.x = mathX;
		zPrev.y = mathY;

		int iter = 1;
		while (iter < _calcIterNum &&
			z.x * z.x + z.y * z.y < 4.0f)
		{
			z = cuCaddf(cuCmulf(zPrev, zPrev), c);
			zPrev = z;

			iter++;
		}
		if (iter >= _calcIterNum)
		{
			PaintTheBelongingPoint(colors, _width, x, y);
		}
		else
		{
			PaintTheNotBelongingPoint(colors, _width, x, y, iter, _calcIterNum, invCalcIterNum);
		}

		index += stride;
		y = index / _width;
		x = index % _width;
	}
}

extern "C" __declspec(dllexport)
__host__ 
void ParallelGPUFractalCalc(uint8_t* colors, int width, int height, float scale, float offsetX, float offsetY, int calcIterNum)
{
	uint8_t* d_colors;
	cudaMalloc(&d_colors, width * height * 4 * sizeof(uint8_t));
	cudaMemcpy(d_colors, colors, width * height * 4 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	int* d_width;
	cudaMalloc(&d_width, sizeof(int));
	cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);

	int* d_height;
	cudaMalloc(&d_height, sizeof(int));
	cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);

	float* d_scale;
	cudaMalloc(&d_scale, sizeof(float));
	cudaMemcpy(d_scale, &scale, sizeof(float), cudaMemcpyHostToDevice);

	float* d_offsetX;
	cudaMalloc(&d_offsetX, sizeof(float));
	cudaMemcpy(d_offsetX, &offsetX, sizeof(float), cudaMemcpyHostToDevice);

	float* d_offsetY;
	cudaMalloc(&d_offsetY, sizeof(float));
	cudaMemcpy(d_offsetY, &offsetY, sizeof(float), cudaMemcpyHostToDevice);

	int* d_calcIterNum;
	cudaMalloc(&d_calcIterNum, sizeof(int));
	cudaMemcpy(d_calcIterNum, &calcIterNum, sizeof(int), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int blocksNum = (width * height * 4 + blockSize - 1) / blockSize;
	FractalCalcOnDevice <<< blocksNum, blockSize >>> (d_colors, d_width, d_height, d_scale, d_offsetX, d_offsetY, d_calcIterNum);

	cudaMemcpy(colors, d_colors, width * height * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(d_calcIterNum);
	cudaFree(d_colors);
	cudaFree(d_height);
	cudaFree(d_offsetX);
	cudaFree(d_offsetY);
	cudaFree(d_scale);
	cudaFree(d_width);
}