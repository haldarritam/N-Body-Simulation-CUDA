#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "timer.h"
#include "aux_declarations_pthreads.h"

pthread_mutex_t mutex_tid;

// main function
int main (const int argc, const char** argv) {  
  int nBodies = 30000;
  float dt = 0.01f; // time step
  int nIters = 10;  // simulation iterations
  int i = 0, iter = 0, rc = 0;

  thread_arg body_ds;
  
  // reading the arguments (argv data)
  switch(argc)
  {
    case 1:
      printf("Default values considered, nBodies: 30000, nIters: 10.\n");
    break;
    case 2:
      nBodies = atoi(argv[1]);
    break;
    case 3:
      nBodies = atoi(argv[1]);
      nIters = atoi(argv[2]);
    break;
    default:
      printf("ERR: Invalid number of arguments passed.\n"
             "Aborting...\n");
      return -1;
    break;
  }

  // initializing body_ds
  body_ds.n = nBodies;
  body_ds.dt = dt;

  int bytes = nBodies*sizeof(bodyStruct); // memory allocation
  bodyStruct *addr = (bodyStruct*)malloc(bytes);
  initialize_bodies(addr, nBodies); // Init mass / pos / vel / acc data 
  body_ds.buf = addr;

  printf("size of struct: %lu\n",sizeof(bodyStruct));
  printf("%f %f %f %f %f %f %f %f %f %f\n",
        addr[0].m, addr[0].x, addr[0].y, addr[0].z, addr[0].ax, addr[0].ay, 
        addr[0].az,addr[0].vx, addr[0].vy, addr[0].vz);


  bytes =  nBodies*sizeof(pthread_t);
  pthread_t *threads = (pthread_t*)malloc(bytes);
  
  double timeStampA = getTimeStamp();
for (iter = 0; iter < nIters; iter++) {
    body_ds.tid = 0;
    for (i = 0; i < nBodies; i++) {
      rc = pthread_create(&threads[i], NULL, 
          nbody_calculation, (void *) &body_ds);
    }

    for (i = 0; i < nBodies; i++) 
      rc = pthread_join(threads[i], NULL);

    for (i = 0 ; i < nBodies; i++) { // integrate position
      body_ds.buf[i].x += body_ds.buf[i].vx*dt;
      body_ds.buf[i].y += body_ds.buf[i].vy*dt;
      body_ds.buf[i].z += body_ds.buf[i].vz*dt;
    }
}

  double timeStampD = getTimeStamp();

  printf("Total interactions: %d\tTotal Time Taken: %lf\n",(nBodies*nBodies*nIters),        (timeStampD - timeStampA));
  printf("%f %f %f %f %f %f %f %f %f %f\n",
        addr[0].m, addr[0].x, addr[0].y, addr[0].z, addr[0].ax, addr[0].ay, 
        addr[0].az,addr[0].vx, addr[0].vy, addr[0].vz);

  free(addr);
  return 0;
}

void initialize_bodies(bodyStruct *b, int n) {
  int i = 0;
  srand(time(0));
  for (i = 0; i < n; i++) {
    b[i].m = 1;
    b[i].x = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].y = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].z = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].ax = 0.0f; 
    b[i].ay = 0.0f;
    b[i].az = 0.0f;
    b[i].vx = 0.0f;
    b[i].vy = 0.0f;
    b[i].vz = 0.0f;
  }
}

void* nbody_calculation(void* arg) {
  
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
    invDist3 = 1/sqrt(distSqr3);

    sx += dx * invDist3; sy += dy * invDist3; sz += dz * invDist3;
  }

  // acceleration calculation
  b->buf[i].ax += sx * b->buf[i].m;
  b->buf[i].ay += sy * b->buf[i].m;
  b->buf[i].az += sz * b->buf[i].m;

  // velocity calculation
  b->buf[i].vx += b->buf[i].ax * b->dt;
  b->buf[i].vy += b->buf[i].ay * b->dt;
  b->buf[i].vz += b->buf[i].az * b->dt;

  return NULL;
}
