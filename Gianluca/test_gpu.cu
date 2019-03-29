#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <cuda_runtime.h>

#define NUM_CPU_THREADS 32
#define MAX_MASS 100.0f
#define MAX_POS_X 1000.0f
#define MAX_POS_Y 1000.0f
#define MAX_VEL_X 0.0f
#define MAX_VEL_Y 0.0f
#define G 8
#define DT 0.0625f
#define DT2 0.00390625f/2
#define DAMPING 1.0f
#define SOFTENING 0.015625f

typedef struct {
	float *m;
	float *r1, *r2;
	float *v1, *v2;
	float *a1, *a2;
	unsigned long nElem, nIter;
} UNIVERSE;

UNIVERSE system;

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// time stamp function in seconds 
double getTimeStamp()
{     
	struct timeval tv;
	gettimeofday (&tv, NULL);
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

void init_MassPositionVelocity ()
{
	// generate different seed for pseudo-random number generator
	time_t t;
	srand ((unsigned int) time(&t));

	// define local variables for convenience
	unsigned long nElem = system.nElem;

	// populating mass, position, & velocity arrays
	unsigned long idx;
	for (idx=0; idx<nElem; idx++) 
	{
		system.m[idx]     = (float) ((double) rand() / (double) (RAND_MAX/MAX_MASS));
		system.r1[2*idx]   = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS_X*2)) - MAX_POS_X);
		system.r1[2*idx+1] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS_Y*2)) - MAX_POS_Y);
		system.v1[2*idx]   = (float) ((double) rand() / (double) (RAND_MAX/(MAX_VEL_X*2)) - MAX_VEL_X);
		system.v1[2*idx+1] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_VEL_Y*2)) - MAX_VEL_Y);
	}
}

void *init_Acceleration_SMT (void *arg)
{
    // define local variables for convenience
    unsigned long start, end, len, offset, nElem;

    nElem = system.nElem;
    offset = (unsigned long) arg;
    len = (unsigned long) system.nElem / NUM_CPU_THREADS;
    start = offset * len;
    end = start + len;

    unsigned long i, j;
    float ax_ip1, ay_ip1;
    float dx_ip1, dy_ip1, rDistSquared, invDistCubed;
    float **i_r = &system.r1;
    float **o_a = &system.a1;

    // calculating NEXT acceleration of each body from the position of every other bodies
    // ... and NEXT velocity of each body utilizing the next acceleration
    for (i=start; i<end; i++)
    {
        ax_ip1 = 0.0;
        ay_ip1 = 0.0;
        for (j=0; j<nElem; j++)
        {
            if (j != i)
            {
                dx_ip1 = *(*i_r + 2*j)   - *(*i_r + 2*i);
                dy_ip1 = *(*i_r + 2*j+1) - *(*i_r + 2*i+1);
                rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + SOFTENING;
                invDistCubed = G*system.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
                ax_ip1 += dx_ip1 * invDistCubed;
                ay_ip1 += dy_ip1 * invDistCubed;
            }
        }

        *(*o_a + 2*i)   = ax_ip1;
        *(*o_a + 2*i+1) = ay_ip1;
    }

    pthread_exit (NULL);
}


__global__ void compute_Device (float *o_r, float *o_v, float *o_a, 
	float *i_r, float *i_v, float *i_a, float *m, const unsigned long nElem)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
}


int main (int argc, char *argv[])
{
	if (argc > 3) {
		printf("Error: Wrong number of arguments.\n");
		exit(EXIT_FAILURE);
	}

	unsigned long nElem = 16000;
	unsigned long nIter = 10;
	char *ptr1, *ptr2;

	if (argc > 1)
		nElem = strtoul(argv[1], &ptr1, 10);
	if (argc > 2)
		nIter = strtoul(argv[2], &ptr2, 10);


	////////////////////////////////////////////////////////////////
	/// SETTING UP DEVICE
	////////////////////////////////////////////////////////////////

	int dev = 0;
	cudaDeviceProp deviceProp;
	checkCudaErrors (cudaGetDeviceProperties (&deviceProp, dev));
	checkCudaErrors (cudaSetDevice (dev));

	printf("\n===== Device Properties ======\n\n");
	printf("  Device %d: %s\n", dev, deviceProp.name);
	printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
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
	printf("  Memory Clock Rate (MHz): %.1f\n", 
		deviceProp.memoryClockRate/1e3);
	printf("  Memory Bus Width (b): %d\n", deviceProp.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %.2f\n\n",
		2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1e6);


	printf("\n===== Simulation Parameters =====\n\n");
	printf("  Number of Bodies = %ld\n", nElem);
	printf("  Number of Time Steps = %ld\n", nIter);
	printf("  Number of CPU Threads = %d\n\n", NUM_CPU_THREADS);
	printf("=================================\n\n");

	////////////////////////////////////////////////////////////////
	/// INITIALIZING SIMULATION
	////////////////////////////////////////////////////////////////

	float *h_m, *h_r1, *h_r2, *h_v1, *h_v2, *h_a1, *h_a2;	// host data
	float *d_m, *d_r1, *d_r2, *d_v1, *d_v2, *d_a1, *d_a2;	// device data

	size_t nBytes = nElem * sizeof(float);
	m    = (float *) malloc(nBytes);
	h_r1 = (float *) malloc(nBytes*2);
	h_r2 = (float *) malloc(nBytes*2);
	h_v1 = (float *) malloc(nBytes*2);
	h_v2 = (float *) malloc(nBytes*2);
	h_a1 = (float *) malloc(nBytes*2);
	h_a2 = (float *) malloc(nBytes*2);

	memset (m, 0, nBytes);
	memset (h_r1, 0, nBytes*2);
	memset (h_r2, 0, nBytes*2);
	memset (h_v1, 0, nBytes*2);
	memset (h_v2, 0, nBytes*2);
	memset (h_a1, 0, nBytes*2);
	memset (h_a2, 0, nBytes*2);

	// initialize data on host size and then transfer to device
	system.m = h_m;
	system.r1 = h_r1;
	system.r2 = h_r2;
	system.v1 = h_v1;
	system.v2 = h_v2;
	system.a1 = h_a1;
	system.a2 = h_a2;
	system.nElem = nElem;
	system.nIter = nIter;

	printf("Initializing bodies on HOST. Time taken: ");
	double time0 = getTimeStamp();
	init_MassPositionVelocity();

	// for portability, explicity create threads in a joinable state
	pthread_t threads [NUM_CPU_THREADS];
	pthread_attr_t attr;
	pthread_attr_init (&attr);
	pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);

	// creating the threads to calculate initial body accelerations on HOST
	for (i=0; i<NUM_CPU_THREADS; i++) {
		rc = pthread_create (&threads[i], &attr, init_Acceleration_SMT, (void *) i);
		if (rc) {
			printf("Error; return code from pthread_create() is %d.\n", rc);
			exit(EXIT_FAILURE);
		}
	}

	// wait on the other threads after initial body accelerations on HOST
	for (i=0; i<NUM_CPU_THREADS; i++) {
		rc = pthread_join (threads[i], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d.\n", rc);
			exit(EXIT_FAILURE);
		}
	}
	printf ("%lfs\n", getTimeStamp()-time0);

	// allocating space in device global memory for data
	checkCudaErrors (cudaMalloc ((void**) &d_m, nBytes));
	checkCudaErrors (cudaMalloc ((void**) &d_r1, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_r2, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_v1, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_v2, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_a1, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_a2, nBytes*2));

	// copying initialized data from host to device
	checkCudaErrors (cudaMemcpy (d_m, h_m, nBytes, cudaMemcopyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r1, h_r1, nBytes*2, cudaMemcopyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r2, h_r2, nBytes*2, cudaMemcopyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v1, h_v1, nBytes*2, cudaMemcopyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v2, h_v2, nBytes*2, cudaMemcopyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_a1, h_a1, nBytes*2, cudaMemcopyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_a2, h_a2, nBytes*2, cudaMemcopyHostToDevice));

	////////////////////////////////////////////////////////////////
	/// PERFORMING SIMULATION ON DEVICE
	////////////////////////////////////////////////////////////////

	dim3 block (1024);
	dim3 grid ((nElem+block.x-1)/(block.x));
	for (unsigned long iter=0; iter<nIter; iter++) {
		if (iter % 2 == 0) {

			cudaDeviceSynchronize ();
		} else {

			cudaDeviceSynchronize ();
		}
	}




	////////////////////////////////////////////////////////////////
	/// SIMULATION COMPLETE
	////////////////////////////////////////////////////////////////

	cudaFree (d_m);
	cudaFree (d_r1); cudaFree (d_r2);
	cudaFree (d_v1); cudaFree (d_v2);
	cudaFree (d_a1); cudaFree (d_a2);

	cudaDeviceReset();

	free (h_m);
	free (h_r1); free (h_r2);
	free (h_v1); free (h_v2);
	free (h_a1); free (h_a2);

	pthread_attr_destroy (&attr);
	pthread_exit(NULL);

	return 0;
}
