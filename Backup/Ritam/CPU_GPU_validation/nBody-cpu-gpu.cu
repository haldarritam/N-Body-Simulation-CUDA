#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <cuda.h>
#include "aux_functions_gpu.h"
#include "aux_declarations_gpu_cpu.h"

pthread_mutex_t mutex_tid;

// main function
int main (const int argc, const char** argv) {
  printf("\n");
  int nBodies = 30000;
  float dt = DT; // time step
  int nIters = 10;  // simulation iterations
  int i = 0, iter = 0, grid_size = 0;
  double total_time_cpu = 0, total_time_gpu = 0;

  thread_arg cpu_body_ds;
  bodyStruct *h_body_ds, *d_body_ds;
  
  // reading the arguments (argv data)
  switch(argc)
  {
    case 1:
      printf("------------------------------------------------------\n\n");
      printf("Default values considered, nBodies: 30000, nIters: 10.\n\n");
      printf("------------------------------------------------------\n\n");
    break;
    case 2:
      nBodies = atoi(argv[1]);
      printf("------------------------------------------------------\n\n");
      printf("Values considered, nBodies: %i, nIters: 10.\n\n", nBodies);
      printf("------------------------------------------------------\n\n");
    break;
    case 3:
      nBodies = atoi(argv[1]);
      nIters = atoi(argv[2]);
      printf("------------------------------------------------------\n\n");
      printf("Values considered, nBodies: %i, nIters: %i.\n\n", nBodies, nIters);
      printf("------------------------------------------------------\n\n");
    break;
    default:
      printf("ERR: Invalid number of arguments passed.\n"
             "Aborting...\n");
      return -1;
  }

  // initializing the cpu data structure
  cpu_body_ds.n = nBodies;
  cpu_body_ds.dt = dt;

  // host side memory allocation for the GPU execution
  size_t bytes = nBodies*sizeof(bodyStruct); // memory allocation
  cudaMallocHost( (bodyStruct **) &h_body_ds, bytes );

  // memory allocation for the CPU execution
  bodyStruct *addr = (bodyStruct*)malloc(bytes);
  cpu_body_ds.buf = addr;

  // initializing the data structures
  initialize_bodies(h_body_ds, addr, nBodies);

  // Device side memory allocation

  cudaMalloc( (bodyStruct **) &d_body_ds, bytes ) ; 

  // determining the grid size
  grid_size = (nBodies+BLOCK_SIZE-1)/BLOCK_SIZE;

  // initializing the dim3 variables

  dim3 block( BLOCK_SIZE, 1, 1 ) ; 
  dim3 grid( grid_size, 1, 1);
  FILE *fp=fopen("out.csv","w");
  
  // starting the iterations
  printf("---------GPU Validation---------\n");
  for (iter = 0; iter < nIters; iter++) {
    
    // print statements
    if(iter%(nIters/3)==0) {
      printf("iter:%i\n",iter);
      printf("MASS 0\t\t\tMASS 1\t\t\tMASS 2\n");
      printf("x:%.04f\t\tx:%.04f\t\tx:%.04f\n",h_body_ds[0].x,h_body_ds[1].x,h_body_ds[2].x);
      printf("y:%.04f\t\ty:%.04f\t\ty:%.04f\n",h_body_ds[0].y,h_body_ds[1].y,h_body_ds[2].y);
      printf("z:%.04f\t\tz:%.04f\t\tz:%.04f\n",h_body_ds[0].z,h_body_ds[1].z,h_body_ds[2].z);
      printf("\n");
    }

    double timeStampA = getTimeStamp();
    // memcopy (host -> device)
    cudaMemcpy( d_body_ds, h_body_ds, bytes, cudaMemcpyHostToDevice  ) ;

    // kernel call
    nbody_calculation_gpu<<<grid, block>>>(d_body_ds, dt, nBodies);
    cudaDeviceSynchronize();

    // memcopy (device -> host)
    cudaMemcpy(h_body_ds, d_body_ds, bytes, cudaMemcpyDeviceToHost);
    
    // integrate and find the new positions
    for (i = 0 ; i < nBodies; i++) { 
      h_body_ds[i].x += h_body_ds[i].vx*dt;
      h_body_ds[i].y += h_body_ds[i].vy*dt;
      h_body_ds[i].z += h_body_ds[i].vz*dt;
    }

    double timeStampD = getTimeStamp();
    gpuErrchk(cudaPeekAtLastError());
    for (i = 0 ; i < nBodies; i++) { 
     fprintf(fp,"%.6f,%.6f,%.6f\n",h_body_ds[i].x, h_body_ds[i].y, h_body_ds[i].z);
    }
    total_time_gpu = total_time_gpu + (timeStampD - timeStampA);
  }
  printf("\n");
  // memory for pthreads
  size_t pBytes = nBodies*sizeof(pthread_t);
  pthread_t *threads = (pthread_t*)malloc(pBytes);

  // CPU multithreaded execution  
  printf("---------CPU Validation---------\n");
  for (iter = 0; iter < nIters; iter++) {
    
    // print statements
    // printf("Iteration: %i\n", iter);
    if(iter%(nIters/3)==0) {
      printf("iter:%i\n",iter);
      printf("MASS 0\t\t\tMASS 1\t\t\tMASS 2\n");
      printf("x:%.04f\t\tx:%.04f\t\tx:%.04f\n",cpu_body_ds.buf[0].x,cpu_body_ds.buf[1].x,cpu_body_ds.buf[2].x);
      printf("y:%.04f\t\ty:%.04f\t\ty:%.04f\n",cpu_body_ds.buf[0].y,cpu_body_ds.buf[1].y,cpu_body_ds.buf[2].y);
      printf("z:%.04f\t\tz:%.04f\t\tz:%.04f\n",cpu_body_ds.buf[0].z,cpu_body_ds.buf[1].z,cpu_body_ds.buf[2].z);
      printf("\n");
    }
    
    double timeStampB = getTimeStamp();
    cpu_body_ds.tid = 0;
    for (i = 0; i < nBodies; i++)
      pthread_create(&threads[i], NULL, nbody_calculation_cpu, (void *) &cpu_body_ds);

    // sync the threads
    for (i = 0; i < nBodies; i++) 
      pthread_join(threads[i], NULL);

    // integrate and find the new positions
    for (i = 0 ; i < nBodies; i++) {
      cpu_body_ds.buf[i].x += cpu_body_ds.buf[i].vx*dt;
      cpu_body_ds.buf[i].y += cpu_body_ds.buf[i].vy*dt;
      cpu_body_ds.buf[i].z += cpu_body_ds.buf[i].vz*dt;
    }
    double timeStampC = getTimeStamp();
    total_time_cpu = total_time_cpu + (timeStampC - timeStampB);
  }
  // fclose(fp);
  // printf statements
  printf("\n");
  printf("CPU -- Total Time Taken: %lf\n", total_time_cpu);
  printf("GPU -- Total Time Taken: %lf\n", total_time_gpu);
  printf("\n");

  // free memory
  free(addr);
  free(threads);
  cudaFreeHost(h_body_ds);
  cudaFree(d_body_ds) ; 
  cudaDeviceReset() ;
  return 0;
}

void initialize_bodies(bodyStruct *b, bodyStruct *c, int n) {
  int i = 0;
  srand(1000);//time(0));
  for (i = 0; i < n; i++) {
    b[i].m = MASS;
    b[i].x = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].y = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].z = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].ax = 0.0f; 
    b[i].ay = 0.0f;
    b[i].az = 0.0f; 
    b[i].vx = 0.0f;
    b[i].vy = 0.0f;
    b[i].vz = 0.0f;
    
    c[i].m =  b[i].m;
    c[i].x =  b[i].x;  
    c[i].y =  b[i].y;  
    c[i].z =  b[i].z;
    c[i].ax = b[i].ax;
    c[i].ay = b[i].ay;
    c[i].az = b[i].az;
    c[i].vx = b[i].vx;
    c[i].vy = b[i].vy;
    c[i].vz = b[i].vz;
  }
}

void* nbody_calculation_cpu(void* arg) {
  
  thread_arg* b = (thread_arg*) arg;

  pthread_mutex_lock(&mutex_tid);
    int i = b->tid;
    b->tid++;
  pthread_mutex_unlock(&mutex_tid); 
  int j = 0;
  float dx = 0.0f, 
        dy = 0.0f,
        dz = 0.0f,
        sx = 0.0f,
        sy = 0.0f,
        sz = 0.0f, 
        distSqr = 0.0f,
        distSqr3 = 0.0f, 
        invDist3 = 0.0f;
  
  for (j = 0; j < b->n; j++) {
    dx = b->buf[j].x - b->buf[i].x;
    dy = b->buf[j].y - b->buf[i].y;
    dz = b->buf[j].z - b->buf[i].z;
    distSqr = dx*dx + dy*dy + dz*dz + EPS;
    distSqr3 = distSqr * distSqr * distSqr;      
    invDist3 = (G * b->buf[j].m)/sqrt(distSqr3);

    sx += dx * invDist3; sy += dy * invDist3; sz += dz * invDist3;
    // printf("i: %i\n", i);
  }

  // acceleration calculation
  b->buf[i].ax += sx;
  b->buf[i].ay += sy;
  b->buf[i].az += sz;

  // velocity calculation
  b->buf[i].vx += b->buf[i].ax * b->dt;
  b->buf[i].vy += b->buf[i].ay * b->dt;
  b->buf[i].vz += b->buf[i].az * b->dt;

  return NULL;
}

__global__ void nbody_calculation_gpu(bodyStruct* b, float dt, int n) {

  int idx = threadIdx.x + blockIdx.x*blockDim.x ; 
  int j = 0;
  float dx = 0.0f, 
        dy = 0.0f,
        dz = 0.0f,
        sx = 0.0f, 
        sy = 0.0f,
        sz = 0.0f,  
        distSqr = 0.0f,
        distSqr3 = 0.0f, 
        invDist3 = 0.0f;
  
  for (j = 0; j < n; j++) {
    dx = b[j].x - b[idx].x;
    dy = b[j].y - b[idx].y;
    dz = b[j].z - b[idx].z;
    distSqr = dx*dx + dy*dy + dz*dz + EPS;
    distSqr3 = distSqr * distSqr * distSqr;      
    invDist3 = (G * b[j].m)/sqrt(distSqr3);

    sx += dx * invDist3; sy += dy * invDist3; sz += dz * invDist3;
  }

  // acceleration calculation
  b[idx].ax += sx;
  b[idx].ay += sy;
  b[idx].az += sz;

  // velocity calculation
  b[idx].vx += b[idx].ax * dt;
  b[idx].vy += b[idx].ay * dt;
  b[idx].vz += b[idx].az * dt;
}
