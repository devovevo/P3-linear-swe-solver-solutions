/**
 * We create a water drop in the middle of the grid. The water drop is a 2D
 * Gaussian function with a maximum height of max_height and a radius of r.
 * In other words, the height of the water drop is given by:
 *
 * h(x, y) = max_height * exp(-((x - center_x)^2 + (y - center_y)^2) / (2 * r^2))
 *
 */
void water_drop(int length, int width, int nx, int ny, double r, double max_height, double *h, double *u, double *v);

/**
 * We create a dam break in the middle of the grid. The dam break is a 2D
 * step function with a maximum height of max_height and a radius of r.
 * In other words, the height of the dam break is given by:
 *
 * h(x, y) = max_height if (x - center_x)^2 + (y - center_y)^2 < r^2 else 1.0
 */
void dam_break(int length, int width, int nx, int ny, double r, double max_height, double *h, double *u, double *v);

/**
 * We create a wave in the middle of the grid. The wave is a sine function
 * with a maximum height of max_height.
 *
 * In other words, the height of the wave is given by:
 *
 * h(x, y) = max_height * sin(2 * PI * x / length)
 */
void wave(int length, int width, int nx, int ny, double max_height, double *h, double *u, double *v);

/**
 * We create a river in the middle of the grid. The river is a constant height
 * of max_height and a constant velocity of 1.0 in the x direction.
 *
 * In other words, the height of the river is given by:
 *
 * h(x, y) = max_height
 */
void river(int length, int width, int nx, int ny, double max_height, double *h, double *u, double *v);