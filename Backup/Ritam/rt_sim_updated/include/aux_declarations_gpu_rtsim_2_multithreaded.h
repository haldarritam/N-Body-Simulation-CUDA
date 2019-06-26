#ifndef __AUX_DECLARATIONS_GPU_CPU_H
#define __AUX_DECLARATIONS_GPU_CPU_H

#define BLOCK_SIZE            1024
#define MASS                  1.0f
#define DT                    0.00012207f
#define EPS                   3.0f
#define G                     800
#define DAMPING	              1.0f

#define SIZE_OF_BODIES        0.9f     // set between -> (0.5 - 3)
#define X_RES                 1920.0f
#define Y_RES                 1080.0f
#define MAX_PIXEL             3
#define MIN_PIXEL             0.5

#define X_WIDTH		      	    300
#define	Y_WIDTH		      	    300
#define	MAX_NUMBER_X	        (int) (X_RES + X_WIDTH)/2
#define	MINIMUM_NUMBER_X      (int) (X_RES - X_WIDTH)/2
#define	MAX_NUMBER_Y	        (int) (Y_RES + Y_WIDTH)/2
#define	MINIMUM_NUMBER_Y      (int) (Y_RES - Y_WIDTH)/2

typedef struct body_pos { 
  float x, y; 
} body_pos;

typedef struct body_dpos { 
  float dx, dy; 
} body_dpos;

typedef struct body_parameters { 
  float m,
        ax, ay, 
        vx, vy; 
} body_parameters;

void initialize_bodies(body_pos*, body_parameters*, int);
inline void GetFrameRate(char*, sf::Clock*);
__global__ void nbody_acc_vel(body_pos*, body_parameters*, float, int);
__global__ void nbody_integration(body_pos*, body_pos*, body_dpos*, body_parameters*, float, int);

#endif
