void init(double *u, double *h, double *v, double length_, double width_, int nx_, int ny_, double H_, double g_, double dt_, int rank_, int num_procs_);

void step();

void free_memory();

void transfer(double *h);