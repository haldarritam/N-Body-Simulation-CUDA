#ifndef HEADER_FILE
#define HEADER_FILE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <cuda_runtime.h>

// global constants
#define NUM_CPU_THREADS 32
#define ND 2				// number of spatial dimensions
#define MIN_MASS 0.1f
#define MAX_MASS 1.0f
#define MAX_POS 1000.0f
#define X_RES 1800.0f
#define Y_RES 900.0f
#define MAX_VEL 8192.0f
#define G 800.0f
#define DT 0.0019531255f		// time step
#define DTd2 0.0009765625f		// (time step) divided by 2
#define DTSQd2 0.00000190734f	// (time step squared) divided by 2
#define DAMPENING 1.0f
#define SOFTENING 1.0f
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// typedef struct {
// 	float4 *r[2];
// 	float4 *v;
// 	float4 *a;
// 	unsigned long nElem, nIter;
// } UNIVERSE;

enum INIT_CONFIG {
	RANDOM_SQUARE_NO_VEL,
	RANDOM_CIRCLE_NO_VEL,
	EXPAND_SHELL,
	SPIRAL_SINGLE_GALAXY,
	SPIRAL_DOUBLE_GALAXY,
	SPIRAL_QUAD_GALAXY,
	NUM_INIT_CONFIGS
};

inline float4 scalevec (float3 v0, float scalar);
inline float normalize (float3 &v0);
inline float dot (float3 v0, float3 v1);
inline float3 cross (float3 v0, float3 v1);
inline float rand_sign ();
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
double getTimeStamp ();
// void print_BodyStats (const float3 *r, const float3 *v, const float3 *a, const unsigned long nElem)
void init_MassPositionVelocity (float4 *r, float4 *v, const unsigned long nElem, const unsigned int config);
// void *init_Acceleration_SMT (void *arg);
void print_simulationParameters (unsigned long nElem, unsigned long nIter, unsigned int cpu_threads);
void print_deviceProperties (int dev, int driverVersion, int runtimeVersion, cudaDeviceProp deviceProp);
__device__ float3 bodyBodyInteraction (float3 ai, float4 bi, float4 bj);
__global__ void initAcceleration (float4 *devA, float4 *devX, const unsigned nTiles);
__device__ float4 calcAcceleration (float4 *devX, unsigned nTiles);
__global__ void calcIntegration (float4 *devX_ip1, const float4 *devX_i,
	float4 *devV_i, float4 *devA_i, const unsigned nElem, const unsigned nTiles);

#endif
