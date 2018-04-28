#pragma once
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" __declspec(dllexport)
__host__ void ParallelGPUFractalCalc(uint8_t* colors, int width, int height, float scale, float offsetX, float offsetY, int calcIterNum);