#include "nbody_helper.h"
#include "nbody_helper_cuda.h"

__global__ void bodyBodyInteraction (float3 ai, float *o_r, float *m, unsigned int j, unsigned int i)
{
	float3 r;
	r.x = o_r[ND*j]   - o_r[ND*i];
	r.y = o_r[ND*j+1] - o_r[ND*i+1];
	r.z = o_r[ND*j+2] - o_r[ND*i+2];
	
	float rDistSquared = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
	float MinvDistCubed = m[j] * rsqrtf(rDistSquared*rDistSquared*rDistSquared);
	
	ai.x = r.x * MinvDistCubed;
	ai.y = r.y * MinvDistCubed;
	ai.z = r.z * MinvDistCubed;
}

__global__ void compute_Device (
	float *o_r, float *o_v, float *o_a, 
	const float *i_r, const float *i_v, const float *i_a, 
	const float *m, const unsigned long nElem)
{
	unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (tid == 0)
		//printf("x: %.2f\ty:%.2f\tz:%.2f\n", i_r[0], i_r[1], i_r[2]);

	float ax_ip1 = 0.0, ay_ip1 = 0.0, az_ip1 = 0.0;
	float dx_ip1, dy_ip1, dz_ip1, rDistSquared, invDistCubed;

	if (tid < nElem) {
		// calculating subsequent position of body (one body per thread)
		o_r[ND*tid]   = i_r[ND*tid]   + i_v[ND*tid]*DT   + i_a[ND*tid]*DTSQd2;		// x-position
		o_r[ND*tid+1] = i_r[ND*tid+1] + i_v[ND*tid+1]*DT + i_a[ND*tid+1]*DTSQd2;	// y-position
		o_r[ND*tid+2] = i_r[ND*tid+2] + i_v[ND*tid+2]*DT + i_a[ND*tid+2]*DTSQd2;	// z-position

		// calculating the NEXT iteration's acceleration and velocity
		#pragma unroll 4
		for (unsigned long j=0; j<nElem; j++) {
			dx_ip1 = o_r[ND*j]   - o_r[ND*tid];
			dy_ip1 = o_r[ND*j+1] - o_r[ND*tid+1];
			dz_ip1 = o_r[ND*j+2] - o_r[ND*tid+2];
			rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
			invDistCubed = m[j] * rsqrtf(rDistSquared*rDistSquared*rDistSquared);
			ax_ip1 += dx_ip1 * invDistCubed;
			ay_ip1 += dy_ip1 * invDistCubed;
			az_ip1 += dz_ip1 * invDistCubed;
		}

		o_a[ND*tid]   = G*ax_ip1;	// x-acceleration
		o_a[ND*tid+1] = G*ay_ip1;	// y-acceleration
		o_a[ND*tid+2] = G*az_ip1;	// z-acceleration

		o_v[ND*tid]   = i_v[ND*tid]   + (i_a[ND*tid]   + ax_ip1)*DTd2;	// x-velocity
		o_v[ND*tid+1] = i_v[ND*tid+1] + (i_a[ND*tid+1] + ay_ip1)*DTd2;	// y-velocity
		o_v[ND*tid+2] = i_v[ND*tid+2] + (i_a[ND*tid+2] + az_ip1)*DTd2;	// z-velocity
	}
}


int main (int argc, char *argv[])
{
	if (argc > 3) {
		printf("Error: Wrong number of arguments.\n");
		exit(EXIT_FAILURE);
	}

	unsigned long nElem = 16384;
	unsigned long nIter = 100;
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
	float *gref_r, *gref_v, *gref_a;

	size_t nBytes = nElem * sizeof(float);
	h_m  = (float *) malloc(nBytes);
	h_r1 = (float *) malloc(nBytes*ND);
	h_r2 = (float *) malloc(nBytes*ND);
	h_v1 = (float *) malloc(nBytes*ND);
	h_v2 = (float *) malloc(nBytes*ND);
	h_a1 = (float *) malloc(nBytes*ND);
	h_a2 = (float *) malloc(nBytes*ND);

	gref_r = (float *) malloc(nBytes*ND);
	gref_v = (float *) malloc(nBytes*ND);
	gref_a = (float *) malloc(nBytes*ND);

	memset (h_m,  0, nBytes);
	memset (h_r1, 0, nBytes*ND);
	memset (h_r2, 0, nBytes*ND);
	memset (h_v1, 0, nBytes*ND);
	memset (h_v2, 0, nBytes*ND);
	memset (h_a1, 0, nBytes*ND);
	memset (h_a2, 0, nBytes*ND);

	memset (gref_r, 0, nBytes*ND);
	memset (gref_v, 0, nBytes*ND);
	memset (gref_a, 0, nBytes*ND);

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
	//print_BodyStats(h_m, h_r1, h_v1, h_a1);

	// allocating space in device global memory for data
	checkCudaErrors (cudaMalloc ((void**) &d_m,  nBytes));
	checkCudaErrors (cudaMalloc ((void**) &d_r1, nBytes*ND));
	checkCudaErrors (cudaMalloc ((void**) &d_r2, nBytes*ND));
	checkCudaErrors (cudaMalloc ((void**) &d_v1, nBytes*ND));
	checkCudaErrors (cudaMalloc ((void**) &d_v2, nBytes*ND));
	checkCudaErrors (cudaMalloc ((void**) &d_a1, nBytes*ND));
	checkCudaErrors (cudaMalloc ((void**) &d_a2, nBytes*ND));

	// copying initialized data from host to device
	checkCudaErrors (cudaMemcpy (d_m,  h_m,  nBytes,   cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r1, h_r1, nBytes*ND, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r2, h_r2, nBytes*ND, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v1, h_v1, nBytes*ND, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v2, h_v2, nBytes*ND, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_a1, h_a1, nBytes*ND, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_a2, h_a2, nBytes*ND, cudaMemcpyHostToDevice));

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
			// cudaMemcpy(gref_m, d_m, nBytes, cudaMemcpyDeviceToHost);
			// cudaMemcpy(gref_r, d_r2, nBytes*2, cudaMemcpyDeviceToHost);
			// cudaMemcpy(gref_v, d_v2, nBytes*2, cudaMemcpyDeviceToHost);
			// cudaMemcpy(gref_a, d_a2, nBytes*2, cudaMemcpyDeviceToHost);

		} else {
			compute_Device <<<grid, block, 0, 0>>> (d_r1, d_v1, d_a1, d_r2, d_v2, d_a2, d_m, nElem);
			cudaDeviceSynchronize ();
			// cudaMemcpy(gref_m, d_m, nBytes, cudaMemcpyDeviceToHost);
			// cudaMemcpy(gref_r, d_r1, nBytes*2, cudaMemcpyDeviceToHost);
			// cudaMemcpy(gref_v, d_v1, nBytes*2, cudaMemcpyDeviceToHost);
			// cudaMemcpy(gref_a, d_a1, nBytes*2, cudaMemcpyDeviceToHost);
		}
		// if (iter%1000 == 0)
		// 	print_BodyStats (gref_m, gref_r, gref_v, gref_a);
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

	free (gref_r);
	free (gref_v);
	free (gref_a);

	pthread_attr_destroy (&attr);
	//pthread_exit(NULL);

	return 0;
}
