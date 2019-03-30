#define BLOCK_SIZE 1024

typedef struct bodyStruct { 
  float m,
        x, y,
        ax, ay, 
        vx, vy;
} bodyStruct;

void initialize_bodies(bodyStruct*, int);
__global__ void nbody_calculation(bodyStruct*, float, int);