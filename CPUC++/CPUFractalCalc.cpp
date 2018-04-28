#include "CPUFractalCalc.h"
#include <omp.h>
#include <complex>
#include <stdint.h>
using namespace std;

uint8_t* g_colors = nullptr;
int g_width = 0;
int g_calcIterNum = 0;
float invCalcIterNum = 0.0f;

enum Bgra32 { B, G, R, A };

void PaintTheBelongingPoint(int x, int y)
{
	for (int j = 0; j < 3; j++)
		g_colors[y * g_width * 4 + x * 4 + j] = 0;
}

void PaintTheNotBelongingPoint(int x, int y, int iter)
{
	g_colors[y * g_width * 4 + x * 4 + Bgra32::B] =
		(255 - (g_calcIterNum - iter) * invCalcIterNum * 255.0f);
	g_colors[y * g_width * 4 + x * 4 + Bgra32::R] =
		(255 - (g_calcIterNum - 0.5f * iter) * invCalcIterNum * 255.0f);
}

extern "C" __declspec(dllexport)
void SequentialCPUFractalCalc(uint8_t* colors, int width, int height, float scale, float offsetX, float offsetY, int calcIterNum)
{
	using namespace std::complex_literals;

	g_colors = colors;
	g_width = width;
	g_calcIterNum = calcIterNum;
	invCalcIterNum = 1.0f / (calcIterNum - 1);

	complex<float> c(0.0f, 0.0f);
	complex<float> z(0.0f, 0.0f);
	complex<float> zPrev(c.real(), c.imag());
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int i = 0; i < 3; i++)
				colors[y * width * 4 + x * 4 + i] = 0;

			colors[y * width * 4 + x * 4 + Bgra32::A] = 255;
		}
	}

	float mathX = 0.0f;
	float mathY = 0.0f;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			mathX = (x - width / 2.0f + offsetX) / scale;
			mathY = (height / 2.0f - y - offsetY) / scale;

			//
			// Begin with the 1-st iteration. (Not with 0-th)
			c = mathX + 1if * mathY;
			z = 0.0f;
			zPrev = mathX + 1if * mathY;

			int iter = 1;
			while (iter < calcIterNum &&
				norm(z) < 4.0f)
			{
				z = zPrev * zPrev + c;
				zPrev = z;

				iter++;
			}
			if (iter >= calcIterNum)
			{
				PaintTheBelongingPoint(x, y);
			}
			else
			{
				PaintTheNotBelongingPoint(x, y, iter);
			}
		}
}
#include <iostream>
extern "C" __declspec(dllexport)
void ParallelCPUFractalCalc(uint8_t* colors, int width, int height, float scale, float offsetX, float offsetY, int calcIterNum)
{
	using namespace std::complex_literals;

	g_colors = colors;
	g_width = width;
	g_calcIterNum = calcIterNum;
	invCalcIterNum = 1.0f / (calcIterNum - 1);


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			for (int i = 0; i < 3; i++)
				colors[y * width * 4 + x * 4 + i] = 0;

			colors[y * width * 4 + x * 4 + Bgra32::A] = 255;
		}
	}

	#pragma omp parallel
	{
		int _width = width;
		int _height = height;
		float _offsetX = offsetX;
		float _offsetY = offsetY;
		float _scale = scale;
		int _calcIterNum = calcIterNum;

		complex<float> c(0.0f, 0.0f);
		complex<float> z(0.0f, 0.0f);
		complex<float> zPrev(c.real(), c.imag());

		float mathX = 0.0f;
		float mathY = 0.0f;

		#pragma omp for
		for (int y = 0; y < _height; ++y)
			for (int x = 0; x < _width; x++)
			{
				mathX = (x - _width / 2.0f + _offsetX) / _scale;
				mathY = (_height / 2.0f - y - _offsetY) / _scale;

				//
				// Begin with the 1-st iteration. (Not with 0-th)
				c = mathX + 1if * mathY;
				z = 0.0f;
				zPrev = mathX + 1if * mathY;

				int iter = 1;
				while (iter < _calcIterNum &&
					norm(z) < 4.0f)
				{
					z = zPrev * zPrev + c;
					zPrev = z;

					iter++;
				}
				if (iter >= _calcIterNum)
					PaintTheBelongingPoint(x, y);
				else
					PaintTheNotBelongingPoint(x, y, iter);
			}
	}
}