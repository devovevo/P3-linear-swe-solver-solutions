#include "scenarios.hpp"

#define PI 3.14159265358979323846

#define h(i, j) h[(i) * (ny + 1) + (j)]
#define u(i, j) u[(i) * (ny + 1) + (j)]
#define v(i, j) v[(i) * (ny + 2) + (j)]

#include <stdlib.h>
#include <math.h>

void water_drop(int length, int width, int nx, int ny, double r, double max_height, double *h, double *u, double *v)
{
    double center_x = length / 2.0;
    double center_y = width / 2.0;

    double drop_r2 = r * r;

    double dx = (double)length / (double)nx;
    double dy = (double)width / (double)ny;

    for (int i = 0; i < nx + 1; i++)
    {
        double x = i * dx + dx / 2.0 - center_x;

        for (int j = 0; j < ny + 1; j++)
        {
            double y = j * dy + dy / 2.0 - center_y;
            double r2 = x * x + y * y;

            h(i, j) = exp(-r2 / (2 * drop_r2)) * max_height;
        }
    }
}

void dam_break(int length, int width, int nx, int ny, double r, double max_height, double *h, double *u, double *v)
{
    double center_x = length / 2.0;
    double center_y = width / 2.0;

    double drop_r2 = r * r;

    double dx = (double)length / (double)nx;
    double dy = (double)width / (double)ny;

    for (int i = 0; i < nx + 1; i++)
    {
        double x = i * dx + dx / 2.0 - center_x;

        for (int j = 0; j < ny + 1; j++)
        {
            double y = j * dy + dy / 2.0 - center_y;
            double r2 = x * x + y * y;

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

void wave(int length, int width, int nx, int ny, double max_height, double *h, double *u, double *v)
{
    double center_x = length / 2.0;
    double center_y = width / 2.0;

    for (int i = 0; i < nx + 1; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            h(i, j) = max_height * sin(2 * PI * i / nx);
        }
    }
}

void river(int length, int width, int nx, int ny, double max_height, double *h, double *u, double *v)
{
    for (int i = 0; i < nx + 1; i++)
    {
        for (int j = 0; j < ny + 1; j++)
        {
            h(i, j) = max_height;
            u(i, j) = 1.0;
        }
    }
}