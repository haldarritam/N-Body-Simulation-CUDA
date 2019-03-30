#define BLOCK_SIZE 1024
#define MASS       100
#define DT         0.001953125f
#define EPS        0.0009765625f
#define G          80

typedef struct bodyStruct { 
  float m, x, y, z,
        ax, ay, az, 
        vx, vy, vz; 
} bodyStruct;

typedef struct thread_arg {
  int tid;     // thread id
  int n;       // number of bodies
  float dt;    // time step
  bodyStruct* buf; //N-Body data structure
} thread_arg;

void initialize_bodies(bodyStruct *, bodyStruct *, int);
void* nbody_calculation_cpu(void*);
__global__ void nbody_calculation_gpu(bodyStruct*, float, int);