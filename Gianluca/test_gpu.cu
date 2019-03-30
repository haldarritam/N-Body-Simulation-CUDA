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
#define DT 0.0019531255f
#define DT2 0.000003814697265625f/2
#define DAMPING 1.0f
#define SOFTENING 0.0009765625f

typedef struct {
	float *m;
	float *r1, *r2;
	float *v1, *v2;
	float *a1, *a2;
	unsigned long nElem, nIter;
} UNIVERSE;

UNIVERSE US;

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
	unsigned long nElem = US.nElem;

	// populating mass, position, & velocity arrays
	unsigned long idx;
	for (idx=0; idx<nElem; idx++) 
	{
		US.m[idx]     = (float) ((double) rand() / (double) (RAND_MAX/MAX_MASS));
		US.r1[2*idx]   = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS_X*2)) - MAX_POS_X);
		US.r1[2*idx+1] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS_Y*2)) - MAX_POS_Y);
		US.v1[2*idx]   = (float) ((double) rand() / (double) (RAND_MAX/(MAX_VEL_X*2)) - MAX_VEL_X);
		US.v1[2*idx+1] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_VEL_Y*2)) - MAX_VEL_Y);
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
    float ax_ip1, ay_ip1;
    float dx_ip1, dy_ip1, rDistSquared, invDistCubed;
    float **i_r = &US.r1;
    float **o_a = &US.a1;

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
                invDistCubed = G*US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
                ax_ip1 += dx_ip1 * invDistCubed;
                ay_ip1 += dy_ip1 * invDistCubed;
            }
        }

        *(*o_a + 2*i)   = ax_ip1;
        *(*o_a + 2*i+1) = ay_ip1;
    }

    pthread_exit (NULL);
}


__global__ void compute_Device (
	float *__restrict__ o_r, float *__restrict__ o_v, float *__restrict__ o_a, 
	const float *__restrict__ i_r, const float *__restrict__ i_v, const float *__restrict__ i_a, 
	const float *__restrict__ m, const unsigned long nElem)
{
	unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == 0)
		printf("x: %.4f\ty:%.4f\n", i_r[0], i_r[1]);

	float ax_ip1 = 0.0, ay_ip1 = 0.0;
	float dx_ip1, dy_ip1, rDistSquared, invDistCubed;

	if (tid < nElem) {
		// calculating subsequent position of body (one body per thread)
		o_r[2*tid]   = __ldg(&i_r[2*tid])   + __ldg(&i_v[2*tid])*DT   + __ldg(&i_a[2*tid])*DT2;		// x-position
		o_r[2*tid+1] = __ldg(&i_r[2*tid+1]) + __ldg(&i_v[2*tid+1])*DT + __ldg(&i_a[2*tid+1])*DT2;	// y-position

		// calculating the NEXT iteration's acceleration and velocity
		for (unsigned long j=0; j<nElem; j++) {
			if (j != tid) {
				dx_ip1 = o_r[2*j]   - o_r[2*tid];
				dy_ip1 = o_r[2*j+1] - o_r[2*tid+1];
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + SOFTENING;
				invDistCubed = G*m[j]*rsqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * invDistCubed;
				ay_ip1 += dy_ip1 * invDistCubed;
			}
		}

		o_a[2*tid]   = ax_ip1;	// x-acceleration
		o_a[2*tid+1] = ay_ip1;	// y-acceleration

		o_v[2*tid]   = i_v[2*tid]   + (i_a[2*tid]   + ax_ip1)*DT/2;	// x-velocity
		o_v[2*tid+1] = i_v[2*tid+1] + (i_a[2*tid+1] + ay_ip1)*DT/2;	// y-velocity
	}
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

	int dev = 0, driverVersion = 0, runtimeVersion = 0;
	cudaDeviceProp deviceProp;
	checkCudaErrors (cudaGetDeviceProperties (&deviceProp, dev));
	checkCudaErrors (cudaSetDevice (dev));
	checkCudaErrors (cudaDriverGetVersion (&driverVersion));
	checkCudaErrors (cudaRuntimeGetVersion (&runtimeVersion));

	printf("\n===== Device Properties ======\n\n");
	printf("  Device %d: %s\n", dev, deviceProp.name);
	printf("  CUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
		driverVersion/1000, (driverVersion%100)/10,
		runtimeVersion/1000, (runtimeVersion%100)/10);
	printf("  CUDA Capability Major/Minor version number: %d.%d\n",
		deviceProp.major, deviceProp.minor);
	printf("  Number of SMs: %d\n", deviceProp.multiProcessorCount);
	printf("  Total amount of global memory: %.2f MB (%llu B)\n",
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


	printf("\n===== Simulation Parameters =====\n\n");
	printf("  Number of Bodies = %ld\n", nElem);
	printf("  Number of Time Steps = %ld\n", nIter);
	printf("  Number of CPU Threads = %d\n\n", NUM_CPU_THREADS);
	printf("=================================\n\n\n");

	////////////////////////////////////////////////////////////////
	/// INITIALIZING SIMULATION
	////////////////////////////////////////////////////////////////

	float *h_m, *h_r1, *h_r2, *h_v1, *h_v2, *h_a1, *h_a2;	// host data
	float *d_m, *d_r1, *d_r2, *d_v1, *d_v2, *d_a1, *d_a2;	// device data

	size_t nBytes = nElem * sizeof(float);
	h_m  = (float *) malloc(nBytes);
	h_r1 = (float *) malloc(nBytes*2);
	h_r2 = (float *) malloc(nBytes*2);
	h_v1 = (float *) malloc(nBytes*2);
	h_v2 = (float *) malloc(nBytes*2);
	h_a1 = (float *) malloc(nBytes*2);
	h_a2 = (float *) malloc(nBytes*2);

	memset (h_m,  0, nBytes);
	memset (h_r1, 0, nBytes*2);
	memset (h_r2, 0, nBytes*2);
	memset (h_v1, 0, nBytes*2);
	memset (h_v2, 0, nBytes*2);
	memset (h_a1, 0, nBytes*2);
	memset (h_a2, 0, nBytes*2);

	// initialize data on host size and then transfer to device
	US.m = h_m;
	US.r1 = h_r1;
	US.r2 = h_r2;
	US.v1 = h_v1;
	US.v2 = h_v2;
	US.a1 = h_a1;
	US.a2 = h_a2;
	US.nElem = nElem;
	US.nIter = nIter;

	printf("Initializing bodies on HOST. Time taken: ");
	double time0 = getTimeStamp();
	init_MassPositionVelocity();

	// for portability, explicity create threads in a joinable state
	pthread_t threads [NUM_CPU_THREADS];
	pthread_attr_t attr;
	pthread_attr_init (&attr);
	pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);

	// creating the threads to calculate initial body accelerations on HOST
	unsigned long i;
	int rc;
	void *status;
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
	checkCudaErrors (cudaMalloc ((void**) &d_m,  nBytes));
	checkCudaErrors (cudaMalloc ((void**) &d_r1, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_r2, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_v1, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_v2, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_a1, nBytes*2));
	checkCudaErrors (cudaMalloc ((void**) &d_a2, nBytes*2));

	// copying initialized data from host to device
	checkCudaErrors (cudaMemcpy (d_m,  h_m,  nBytes,   cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r1, h_r1, nBytes*2, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r2, h_r2, nBytes*2, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v1, h_v1, nBytes*2, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v2, h_v2, nBytes*2, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_a1, h_a1, nBytes*2, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_a2, h_a2, nBytes*2, cudaMemcpyHostToDevice));

	////////////////////////////////////////////////////////////////
	/// PERFORMING SIMULATION ON DEVICE
	////////////////////////////////////////////////////////////////

	dim3 block (1024);
	dim3 grid  ((nElem+block.x-1)/(block.x));

	double timestamp_GPU_start = getTimeStamp();
	for (unsigned long iter=0; iter<nIter; iter++) {
		if (iter % 2 == 0) {
			compute_Device <<<grid, block, 0, 0>>> (d_r2, d_v2, d_a2, d_r1, d_v1, d_a1, d_m, nElem);
			cudaDeviceSynchronize ();
		} else {
			compute_Device <<<grid, block, 0, 0>>> (d_r1, d_v1, d_a1, d_r2, d_v2, d_a2, d_m, nElem);
			cudaDeviceSynchronize ();
		}
	}
	double timestamp_GPU_end = getTimeStamp();
	double elapsedTime = timestamp_GPU_end - timestamp_GPU_start;

	////////////////////////////////////////////////////////////////
	/// SIMULATION COMPLETE
	////////////////////////////////////////////////////////////////

	cudaFree (d_m);
	cudaFree (d_r1); cudaFree (d_r2);
	cudaFree (d_v1); cudaFree (d_v2);
	cudaFree (d_a1); cudaFree (d_a2);

	checkCudaErrors (cudaDeviceReset());
	printf("Device successfully reset.\n");
	printf("\nElapsed Time: %lfs\n", elapsedTime);
	printf("Average timestep simulation duration: %lfs\n", elapsedTime/nIter); 


	free (h_m);
	free (h_r1); free (h_r2);
	free (h_v1); free (h_v2);
	free (h_a1); free (h_a2);

	pthread_attr_destroy (&attr);
	//pthread_exit(NULL);

	return 0;
}
