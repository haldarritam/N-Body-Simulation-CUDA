#ifndef __AUX_DECLARATIONS_GPU_CPU_H
#define __AUX_DECLARATIONS_GPU_CPU_H

#define BLOCK_SIZE            1024
#define MASS                  1
#define DT                    0.001953125f
#define EPS                   0.0009765625f
#define G                     1

#define SIZE_OF_BODIES        0.7f     // (0.5 - 3)
#define X_RES                 1900.0f
#define Y_RES                 1080.0f
#define MAX_PIXEL             3
#define MIN_PIXEL             0.5

typedef struct body_pos { 
  float x, y, z; 
} body_pos;

typedef struct body_parameters { 
  float m,
        ax, ay, az, 
        vx, vy, vz; 
} body_parameters;

void initialize_bodies(body_pos*, body_parameters*, int);
__global__ void nbody_acc_vel(body_pos*, body_parameters*, float, int);
__global__ void nbody_integration(body_pos*, body_parameters*, float, int);

#endif