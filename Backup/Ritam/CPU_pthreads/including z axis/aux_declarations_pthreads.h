#define EPS 1e-9f

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

void initialize_bodies(bodyStruct*, int);
void* nbody_calculation(void*);