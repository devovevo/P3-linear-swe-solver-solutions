#include <stdlib.h>
#include <math.h>

#include "scenarios.hpp"

#include "common.hpp"

#define PI 3.14159265358979323846

void water_drop(int size_x, int size_y, int nx, int ny, float r, float max_height, float *h, float *u, float *v)
{
    float center_x = size_x / 2.0;
    float center_y = size_y / 2.0;

    float drop_r2 = r * r;

    float dx = (float)size_x / nx;
    float dy = (float)size_y / ny;

    for (int i = 0; i < nx; i++)
    {
        float x = i * dx + dx / 2.0 - center_x;

        for (int j = 0; j < ny; j++)
        {
            float y = j * dy + dy / 2.0 - center_y;
            float r2 = x * x + y * y;

            h(i, j) = exp(-r2 / (2 * drop_r2)) * max_height;
        }
    }
}

void dam_break(int length, int width, int nx, int ny, float r, float max_height, float *h, float *u, float *v)
{
    float center_x = length / 2.0;
    float center_y = width / 2.0;

    float drop_r2 = r * r;

    float dx = (float)length / (float)nx;
    float dy = (float)width / (float)ny;

    for (int i = 0; i < nx; i++)
    {
        float x = i * dx + dx / 2.0 - center_x;

        for (int j = 0; j < ny; j++)
        {
            float y = j * dy + dy / 2.0 - center_y;
            float r2 = x * x + y * y;

            if (r2 < drop_r2)
            {
                h(i, j) = max_height;
            }
            else
            {
                h(i, j) = 1.0;
            }
        }
    }
}

void wave(int length, int width, int nx, int ny, float max_height, float *h, float *u, float *v)
{
    float center_x = length / 2.0;
    float center_y = width / 2.0;

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            h(i, j) = max_height * sin(2 * PI * i / nx);
        }
    }
}

void river(int length, int width, int nx, int ny, float max_height, float *h, float *u, float *v)
{
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            h(i, j) = max_height;
            u(i, j) = 1.0;
        }
    }
}