#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// global constants
#define NUM_CPU_THREADS 32
#define ND 3				// number of spatial dimensions
#define MAX_MASS 100.0f
#define MAX_POS 10000.0f
#define MAX_VEL 8192.0f
#define G 16384
#define DT 0.0019531255f
#define DTd2 0.0009765625f
#define DTSQd2 0.00000190734f
#define DAMPENING 1.0f
#define SOFTENING 1.0f


typedef struct {
	float4 *r[2];
	float4 *v;
	float4 *a;
	unsigned long nElem, nIter;
} UNIVERSE;

UNIVERSE US;

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval tv;
	gettimeofday (&tv, NULL);
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}


void print_BodyStats (const float *m, const float *r, const float *v, const float *a)
{
    unsigned long nElem = US.nElem;

    printf("\n");
    // print body number
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("Mass %ld\n", idx);
        else
            printf("Mass %ld\t", idx);
    }

    // print Mass
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n", m[idx]);
        else
            printf("%.2f\t", m[idx]);
    }

	// print position
	for (unsigned int dim=0; dim<ND; dim++) {
		for (unsigned long idx=0; idx<nElem; idx++) {
			if (idx == nElem-1)
				printf("%.2f\n", r[ND*idx + dim]);
			else
				printf("%.2f\t", r[ND*idx + dim]);
		}
	}	
	
	// print velocity
	for (unsigned int dim=0; dim<ND; dim++) {
		for (unsigned long idx=0; idx<nElem; idx++) {
			if (idx == nElem-1)
				printf("%.2f\n", v[ND*idx + dim]);
			else
				printf("%.2f\t", v[ND*idx + dim]);
		}
	}	

	// print acceleration
	for (unsigned int dim=0; dim<ND; dim++) {
		for (unsigned long idx=0; idx<nElem; idx++) {
			if (idx == nElem-1)
				printf("%.2f\n", a[ND*idx + dim]);
			else
				printf("%.2f\t", a[ND*idx + dim]);
		}
	}	
}


void init_MassPositionVelocity ()
{
	// generate different seed for pseudo-random number generator
	// time_t t;
	// srand ((unsigned int) time(&t));
	srand ((unsigned int) 1000);

	// define local variables for convenience
	unsigned long nElem = US.nElem;

	// populating mass, position, & velocity arrays
	unsigned long idx;
	unsigned int dim;

	float rx, ry, rz, mag_cross_sq;
	for (idx=0; idx<nElem; idx++) 
	{
		// initializing mass and position
		if (idx % 100 == 0) {
			US.m[idx] = 128000*MAX_MASS;
			for (dim=0; dim<ND; dim++) {
				US.r1[ND*idx+dim] = (float) ((double) rand() / (double) (RAND_MAX/(1000*2)) - 1000);
			}
		} else {
			US.m[idx] = (float) ((double) rand() / (double) (RAND_MAX/MAX_MASS));
			for (dim=0; dim<ND; dim++) {
				US.r1[ND*idx+dim] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS*2)) - MAX_POS);
			}
		}

		// calculating cross product along z-plane for rotational initial velocity about k-hat
		if (idx % 100 == 0) {
			US.v1[ND*idx + 0] = 0;
			US.v1[ND*idx + 1] = 0;
			US.v1[ND*idx + 2] = 0;
		} else {
			rx = US.r1[ND*idx + 0]; ry = US.r1[ND*idx + 1]; rz = US.r1[ND*idx + 2];
			mag_cross_sq = sqrtf(rx*rx + ry*ry + rz*rz);
			US.v1[ND*idx + 0] =    MAX_VEL * ry/mag_cross_sq;
			US.v1[ND*idx + 1] = -1*MAX_VEL * rx/mag_cross_sq;
			US.v1[ND*idx + 2] = 0.0;
		}
	}
}

void *init_Acceleration_SMT (void *arg)
{
	// define local variables for convenience
	unsigned long start, end, len, offset, nElem;

	nElem = US.nElem;
	offset = (unsigned long) arg;
	len = (unsigned long) US.nElem / NUM_CPU_THREADS;
	start = offset * len;
	end = start + len;

	unsigned long i, j;
	float ax_ip1, ay_ip1, az_ip1;
	float dx_ip1, dy_ip1, dz_ip1, rDistSquared, MinvDistCubed;
	float **i_r = &(US.r1);
	float **o_a = &(US.a1);

	// calculating NEXT acceleration of each body from the position of every other bodies
	// ... and NEXT velocity of each body utilizing the next acceleration
	for (i=start; i<end; i++) {
		ax_ip1 = 0.0;
		ay_ip1 = 0.0;
		az_ip1 = 0.0;
		for (j=0; j<nElem; j++) {
			dx_ip1 = *(*i_r + (ND*j+0)) - *(*i_r + (ND*i+0));
			dy_ip1 = *(*i_r + (ND*j+1)) - *(*i_r + (ND*i+1));
			dz_ip1 = *(*i_r + (ND*j+2)) - *(*i_r + (ND*i+2));
			rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
			MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
			ax_ip1 += dx_ip1 * MinvDistCubed;
			ay_ip1 += dy_ip1 * MinvDistCubed;
			az_ip1 += dz_ip1 * MinvDistCubed;
		}

		*(*o_a + (ND*i+0)) = G*ax_ip1;
		*(*o_a + (ND*i+1)) = G*ay_ip1;
		*(*o_a + (ND*i+2)) = G*az_ip1;
	}

	pthread_exit (NULL);
}

// HELPER FUNCTIONS
inline float4 scalevec (float3 v0, float scalar)
{
	float4 rt = v0;
	rt.x *= scalar;
	rt.y *= scalar;
	rt.z *= scalar;
	return rt;
}

inline float normalize (float3 &v0)
{
	float dist = sqrtf(dot(v0, v0));
	if (dist > 1e-6) {
		v0.x /= dist;
		v0.y /= dist;
		v0.z /= dist;
	} else {
		v0.x *= 1e6;
		v0.y *= 1e6;
		v0.z *= 1e6;
	}
	
	return dist;
}

inline float dot (float3 v0, float3 v1)
{
	return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

inline float3 cross (float3 v0, float3 v1)
{
	float3 v2;
	v2.x = v0.y*v1.z - v0.z*v1.y;
	v2.y = v0.z*v1.x - v0.x*v1.z;
	v2.z = v0.z*v1.y - v0.y*v1.x;
	return v2;
}

void print_simulationParameters (unsigned long nElem, unsigned long nIter, unsigned int cpu_threads)
{
	printf("\n===== Simulation Parameters =====\n\n");
	printf("  Number of Bodies = %ld\n", nElem);
	printf("  Number of Time Steps = %ld\n", nIter);
	printf("  Number of CPU Threads = %d\n\n", cpu_threads);
	printf("=================================\n\n\n");
}


void print_deviceProperties (int dev, int driverVersion, int runtimeVersion, cudaDeviceProp deviceProp)
{
	printf("\n===== Device Properties ======\n\n");
	printf("  Device %d: %s\n", dev, deviceProp.name);
	printf("  CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
		driverVersion/1000, (driverVersion%100)/10,
		runtimeVersion/1000, (runtimeVersion%100)/10);
	printf("  CUDA Capability Major/Minor version number: %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
	printf("  Total amount of global memory: %.2f GB (%llu B)\n",
		(float) deviceProp.totalGlobalMem/pow(1024.0,3),
		(unsigned long long) deviceProp.totalGlobalMem);
	printf("  Total amount of constant memory: %4.2f kB\n", 
		deviceProp.totalConstMem/1024.0);
	printf("  Total amount of shared memory per block: %4.2f kB\n",
		deviceProp.sharedMemPerBlock/1024.0);
	printf("  Total number of registers available per block: %d\n",
		deviceProp.regsPerBlock);
	printf("  Warp size: %d\n", deviceProp.warpSize);
	printf("  Maximum number of threads per block: %d\n",
		deviceProp.maxThreadsPerBlock);
	printf("  Maximum number of threads per SM: %d\n",
		deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of warps per SM: %d\n",
		deviceProp.maxThreadsPerMultiProcessor/32);
	printf("  Maximum size of each block dimension: %d x %d x %d\n",
		deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	printf("  Maximum size of each grid dimension: %d x %d x %d\n",
		deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch: %lu B\n", deviceProp.memPitch);
	printf("  Memory Clock Rate (MHz): %.1f\n", 
		deviceProp.memoryClockRate/1e3);
	printf("  Memory Bus Width (b): %d\n", deviceProp.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %.2f\n\n",
		2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1e6);
}


