#ifndef HEADER_FILE
#define HEADER_FILE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <SFML/Graphics.hpp>

// global constants
#define NUM_CPU_THREADS 32
#define BLOCK_SIZE 1024
#define ND 2				// number of spatial dimensions
#define MIN_MASS 0.1f
#define MAX_MASS 1.0f
#define MAX_POS 1000.0f
#define X_RES 1920.0f
#define Y_RES 1080.0f
#define SIZE_OF_BODIES 0.9
#define MAX_VEL 8192.0f
#define G 80.0f
#define DT 0.0019531255f		// time step
#define DTd2 0.0009765625f		// (time step) divided by 2
#define DTSQd2 0.00000190734f	// (time step squared) divided by 2
#define DAMPENING 1.0f
#define SOFTENING 1.0f
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*typedef struct {
	float3 *r[2];
	float3 *v;
	float3 *a;
	unsigned int nElem, nIter;
} UNIVERSE;

typedef struct {
	unsigned int tid;
	struct UNIVERSE *system;
} THREAD_STRUCT;*/

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
inline float dot (float2 v0, float2 v1);
inline float3 cross (float3 v0, float3 v1);
inline float rand_sign ();
inline void GetFrameRate(char* char_buffer, sf::Clock* clock);
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort);
double getTimeStamp ();
// void print_BodyStats (const float3 *r, const float3 *v, const float3 *a, const unsigned long nElem)
void init_MassPositionVelocity (float3 *r, float3 *v, const unsigned long nElem, const unsigned int config);
// void *init_Acceleration_SMT (void *arg);
void print_simulationParameters (unsigned long nElem, unsigned long nIter, unsigned int cpu_threads);
void print_deviceProperties (int dev, int driverVersion, int runtimeVersion, cudaDeviceProp deviceProp);
__device__ float2 bodyBodyInteraction (float2 ai, const float3 bi, const float3 bj);
__global__ void initAcceleration (float3 *devA, const float3 *__restrict__ devX, const unsigned nTiles);
__device__ float3 calcAcceleration (const float3 *__restrict__ devX, const unsigned nTiles);
__global__ void calcIntegration (float3 *devX_ip1, const float3 *__restrict__ devX_i,
	float3 *devV_i, float3 *devA_i, const unsigned nElem, const unsigned nTiles);
void *computeHost_SMT (void *arg);

#endif

