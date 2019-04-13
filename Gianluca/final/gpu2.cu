#include "nbody_helper2.h"

int main (int argc, char *argv[])
{
	if (argc > 4) {
		printf("Error: Wrong number of arguments.\n");
		exit(EXIT_FAILURE);
	}

	unsigned long nElem = 16384;
	unsigned long nIter = 100;
	unsigned int config = 0;
	char *ptr1, *ptr2, *ptr3;

	// acquiring command line arguments, if any.
	if (argc > 1)	// no. of elements
		nElem  = (unsigned int) strtoul(argv[1], &ptr1, 10);
	if (argc > 2)	// no. of iterations
		nIter  = (unsigned int) strtoul(argv[2], &ptr2, 10);
	if (argc > 3)	// initial config of bodies
		config = (unsigned int) strtoul(argv[3], &ptr3, 10);


	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// SETTING UP DEVICE
	/////////////////////////////////////////////////////////////////////////////////////////////////

	int dev = 0, driverVersion = 0, runtimeVersion = 0;
	cudaDeviceProp deviceProp;
	checkCudaErrors (cudaGetDeviceProperties (&deviceProp, dev));
	checkCudaErrors (cudaSetDevice (dev));
	checkCudaErrors (cudaDriverGetVersion (&driverVersion));
	checkCudaErrors (cudaRuntimeGetVersion (&runtimeVersion));

	print_deviceProperties (dev, driverVersion, runtimeVersion, deviceProp);
	print_simulationParameters (nElem, nIter, NUM_CPU_THREADS);

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// INITIALIZING SIMULATION
	/////////////////////////////////////////////////////////////////////////////////////////////////

	float3 *h_dref_r, *h_dref_v;
	float3 *d_r[2],   *d_v,      *d_a;
	
	size_t nBytes = nElem * sizeof(float3);

	// allocating page-locked memory for higher communication bandwidth during real-time vis.
	checkCudaErrors (cudaMallocHost ((void**) &h_dref_r, nBytes));
	checkCudaErrors (cudaMallocHost ((void**) &h_dref_v, nBytes));

	checkCudaErrors (cudaMalloc ((void**) &(d_r[0]), nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_r[1]), nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_v),    nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_a),    nBytes));


	printf("Initializing bodies' positions / velocities on HOST. Time taken: ");
	double time0 = getTimeStamp();
	init_MassPositionVelocity(h_dref_r, h_dref_v, nElem, config);
	printf ("%lfs\n", getTimeStamp()-time0);
	//print_BodyStats(h_m, h_r1, h_v1, h_a1);

	// setting shmem and L1 cache config. 
	// 		cudaFuncCachePreferNone:	no preference (default)
	//		cudaFuncCachePreferShared:	prefer 48kB shared memory and 16kB L1 cache
	//		cudaFuncCachePreferL1:		prefer 48kB L1 cache and 16kB shmem
	//		cudaFuncCachePreferEqual:	prefer 32kB L1 cache and 32kB shmem
	cudaFuncCache cacheConfig = cudaFuncCachePreferNone;
	checkCudaErrors (cudaDeviceSetCacheConfig (cacheConfig));

	// copying initialized data from host to device
	checkCudaErrors (cudaMemcpy (d_r[0], h_dref_r, nBytes, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v,    h_dref_v, nBytes, cudaMemcpyHostToDevice));

	// compute initial acceleration of bodies on device
	dim3 block_size (1024);
	dim3 grid_size	((nElem + block_size.x-1)/(block_size.x));
	unsigned int nTiles = (nElem + block_size.x-1)/block_size.x;
	printf("Computing initial acceleration on device. Time Taken: ");
	time0 = getTimeStamp();
	initAcceleration <<<grid_size, block_size, 0, 0>>> (d_a, d_r[0], nTiles);
	cudaDeviceSynchronize ();
	printf ("%lfs\n", getTimeStamp()-time0);

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// PERFORMING SIMULATION ON DEVICE
	/////////////////////////////////////////////////////////////////////////////////////////////////

	printf("Computing positions on device. Time taken: ");
	double timestamp_GPU_start = getTimeStamp();
	for (unsigned iter=0; iter<nIter; iter++) {
		calcIntegration <<<grid_size, block_size, 0, 0>>> (
			d_r[(iter+1)%2],	// pointer to new positions
			d_r[iter%2], 		// pointer to curr positions
			d_v, 				// pointer to curr velocities
			d_a, 				// pointer to curr accelerations
			nElem, 				// number of bodies in simulation
			nTiles);			// number of shared memory sections per block
		
		cudaDeviceSynchronize ();
		// cudaMemcpy(h_dref_r, d_r[(iter+1)%2], nBytes, cudaMemcpyDeviceToHost);

		// if (iter%1000 == 0)
		// 	print_BodyStats (gref_m, gref_r, gref_v, gref_a);
	}

	double timestamp_GPU_end = getTimeStamp();
	double elapsedTime = timestamp_GPU_end - timestamp_GPU_start;
	printf("%.6lfs\n", elapsedTime);
	printf("Elapsed Time per Iteration: %.6lfs\n", elapsedTime/nIter);

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// SIMULATION COMPLETE -- (free memory)
	/////////////////////////////////////////////////////////////////////////////////////////////////

	printf("Freeing global and system memory.\n");

	// free global memory on GPU DRAM
	checkCudaErrors (cudaFree (d_r[0]));
	checkCudaErrors (cudaFree (d_r[1]));
	checkCudaErrors (cudaFree (d_v));
	checkCudaErrors (cudaFree (d_a));
	
	// free page-locked ("pinned") memory on system DRAM
	checkCudaErrors (cudaFreeHost (h_dref_r));
	checkCudaErrors (cudaFreeHost (h_dref_v));

	checkCudaErrors (cudaDeviceReset());
	
	printf("Device successfully reset.\n");
	printf("\nElapsed Time: %lfs\n", elapsedTime);
	printf("Average timestep simulation duration: %lfs\n", elapsedTime/nIter); 

	return 0;
}
