#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <cuda.h>
#include "aux_functions_gpu.h"
#include "aux_declarations_gpu.h"

// main function
int main (const int argc, const char** argv) {  
  int nBodies = 30000;
  float dt = 0.01f; // time step
  int nIters = 10;  // simulation iterations
  int i = 0, iter = 0, grid_size = 0;

  bodyStruct *h_body_ds, *d_body_ds;
  
  // reading the arguments (argv data)
  switch(argc)
  {
    case 1:
      printf("Default values considered, nBodies: 30000, nIters: 10.\n");
    break;
    case 2:
      nBodies = atoi(argv[1]);
      printf("Values considered, nBodies: %i, nIters: 10.\n", nBodies);
    break;
    case 3:
      nBodies = atoi(argv[1]);
      nIters = atoi(argv[2]);
      printf("Values considered, nBodies: %i, nIters: %i.\n", nBodies, nIters);
    break;
    default:
      printf("ERR: Invalid number of arguments passed.\n"
             "Aborting...\n");
      return -1;
    break;
  }


  int bytes = nBodies*sizeof(bodyStruct); // memory allocation
  cudaMallocHost( (bodyStruct **) &h_body_ds, bytes );
  initialize_bodies(h_body_ds, nBodies); // Init mass / pos / vel / acc data 

  printf("%f %f %f %f %f %f %f\n",
        h_body_ds[0].m, h_body_ds[0].x, h_body_ds[0].y, h_body_ds[0].ax, h_body_ds[0].ay, 
        h_body_ds[0].vx, h_body_ds[0].vy);

  // Device side memory allocation

  cudaMalloc( (bodyStruct **) &d_body_ds, bytes ) ; 

  // determining the grid size
  grid_size = ceil (nBodies / BLOCK_SIZE);
  
  double timeStampA = getTimeStamp();

  // initializing the dim3 variables

  dim3 block( BLOCK_SIZE, 1, 1 ) ; 
  dim3 grid( grid_size, 1, 1);

  // starting the iterations
  for (iter = 0; iter < nIters; iter++) {

    // memcopy (host -> device)
  cudaMemcpy( d_body_ds, h_body_ds, bytes, cudaMemcpyHostToDevice  ) ;

    // kernel call
    nbody_calculation<<<grid, block>>>(d_body_ds, dt, nBodies);
    cudaDeviceSynchronize();

    // memcopy (device -> host)
    cudaMemcpy(h_body_ds, d_body_ds, bytes, cudaMemcpyDeviceToHost);
    // integrate and find the new positions
    for (i = 0 ; i < nBodies; i++) { 
      h_body_ds[i].x += h_body_ds[i].vx*dt;
      h_body_ds[i].y += h_body_ds[i].vy*dt;
    }
  }
  double timeStampD = getTimeStamp();

    // printf statements
  printf("%f %f %f %f %f %f %f\n",
        h_body_ds[0].m, h_body_ds[0].x, h_body_ds[0].y, h_body_ds[0].ax, h_body_ds[0].ay, 
        h_body_ds[0].vx, h_body_ds[0].vy);
  
  printf("Total interactions: %li\tTotal Time Taken: %lf\n",
  (long)(nBodies*nBodies*nIters),(timeStampD - timeStampA));

  // free memory
  cudaFreeHost( h_body_ds );
  cudaFree( d_body_ds   ) ; 
  cudaDeviceReset() ;
  return 0;
}

void initialize_bodies(bodyStruct *b, int n) {
  int i = 0;
  srand(time(0));
  for (i = 0; i < n; i++) {
    b[i].m = 1;
    b[i].x = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].y = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].ax = 0.0f; 
    b[i].ay = 0.0f;
    b[i].vx = 0.0f;
    b[i].vy = 0.0f;
  }
}

__global__ void nbody_calculation(bodyStruct* b, float dt, int n) {

  int idx = threadIdx.x + blockIdx.x*blockDim.x ; 
  float eps = 1e-9f;
  int j = 0;
  float dx = 0.0f, 
        dy = 0.0f,
        sx = 0.0f, 
        sy = 0.0f, 
        distSqr = 0.0f,
        distSqr3 = 0.0f, 
        invDist3 = 0.0f;
  
  for (j = 0; j < n; j++) {
    dx = b[j].x - b[idx].x;
    dy = b[j].y - b[idx].y;
    distSqr = dx*dx + dy*dy + eps;
    distSqr3 = distSqr * distSqr * distSqr;      
    invDist3 = 1/sqrt(distSqr3);

    sx += dx * invDist3; sy += dy * invDist3;
  }

  // acceleration calculation
  b[idx].ax += sx * b[idx].m;
  b[idx].ay += sy * b[idx].m;

  // velocity calculation
  b[idx].vx += b[idx].ax * dt;
  b[idx].vy += b[idx].ay * dt;
}
