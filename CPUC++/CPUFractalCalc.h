#pragma once
#include <stdint.h>

extern "C"
{
	__declspec(dllexport)
		void SequentialCPUFractalCalc(uint8_t* colors, int width, int height, float scale, float offsetX, float offsetY, int calcIterNum);
	__declspec(dllexport)
		void ParallelCPUFractalCalc(uint8_t* colors, int width, int height, float scale, float offsetX, float offsetY, int calcIterNum);
}