#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
#include "aux_declarations.h"

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  float dt = 0.01f; // time step
  int nIters = 10;  // simulation iterations
  int i = 0, j = 0;
  
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

  int bytes = nBodies*sizeof(bodyStruct);
  bodyStruct *buf = (bodyStruct*)malloc(bytes);

  initialize_bodies(buf, nBodies); // Init mass / pos / vel / acc data 

  double timeStampA = getTimeStamp();

  for (i = 0; i < nIters; i++) {
    double timeStampB = getTimeStamp();

    cal_acceleration(buf, nBodies); // compute acceleration
    cal_velocity(buf, nBodies, dt); // compute velocity

    for (j = 0 ; j < nBodies; j++) { // integrate position
      buf[j].x += buf[j].vx*dt;
      buf[j].y += buf[j].vy*dt;
      buf[j].z += buf[j].vz*dt;
    }
    double timeStampC = getTimeStamp();
    printf("Iter: %i\tTime Taken: %lf\n",i,(timeStampC - timeStampB));
  }
  double timeStampD = getTimeStamp();

  printf("Total interactions: %d\tTotal Time Taken: %lf\n",(nBodies*nBodies*nIters),        (timeStampD - timeStampA));

  free(buf);
}

void initialize_bodies(bodyStruct *b, int n) {
  int i = 0;
  for (i = 0; i < n; i++) {
    b[i].m = 1;
    b[i].x = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].y = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].z = 2.0f * ((rand() / (float)RAND_MAX) * 100.0f) - 100.0f;
    b[i].ax = 0; 
    b[i].ay = 0;
    b[i].az = 0;
    b[i].vx = 0;
    b[i].vy = 0;
    b[i].vz = 0;
  }
}

void cal_acceleration(bodyStruct *b, int n) {
  int i = 0, j = 0;
  float dx = 0.0f, 
        dy = 0.0f, 
        dz = 0.0f, 
        sx = 0.0f, 
        sy = 0.0f, 
        sz = 0.0f, 
        distSqr = 0.0f,
        distSqr3 = 0.0f, 
        invDist3 = 0.0f;
  for (i = 0; i < n; i++) {
    sx = 0.0f; 
    sy = 0.0f;
    sz = 0.0f;
    for (j = 0; j < n; j++) {
      dx = b[j].x - b[i].x;
      dy = b[j].y - b[i].y;
      dz = b[j].z - b[i].z;
      distSqr = dx*dx + dy*dy + dz*dz + EPS;
      distSqr3 = distSqr * distSqr * distSqr;      
      invDist3 = sqrt(distSqr3);

      sx += dx * invDist3; sy += dy * invDist3; sz += dz * invDist3;
    }

    b[i].ax += sx * b[i].m;
    b[i].ay += sy * b[i].m;
    b[i].az += sz * b[i].m;
  }
}

void cal_velocity(bodyStruct *b, int n, float dt) {
  int i = 0;
  for (i = 0; i < n; i++) {
    b[i].vx += b[i].ax * dt;
    b[i].vy += b[i].ay * dt;
    b[i].vz += b[i].az * dt;
  }
}
