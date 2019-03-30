#define BLOCK_SIZE 1024
#define MASS       1
#define DT         0.001
#define EPS        1e-9f
#define G          10

typedef struct bodyStruct { 
  float m, x, y, z,
        ax, ay, az, 
        vx, vy, vz; 
} bodyStruct;

void initialize_bodies(bodyStruct*, int);
__global__ void nbody_calculation(bodyStruct*, float, int);