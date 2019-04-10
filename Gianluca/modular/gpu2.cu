#include "nbody_helper.h"
#include "nbody_helper_cuda.h"

//__global__ void bodyBodyInteraction (float3 ai, float *o_r, float *m, unsigned int j, unsigned int i)
//{
//	float3 r;
//	r.x = o_r[ND*j]   - o_r[ND*i];
//	r.y = o_r[ND*j+1] - o_r[ND*i+1];
//	r.z = o_r[ND*j+2] - o_r[ND*i+2];
//	
//	float rDistSquared = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
//	float MinvDistCubed = m[j] * rsqrtf(rDistSquared*rDistSquared*rDistSquared);
//	
//	ai.x = r.x * MinvDistCubed;
//	ai.y = r.y * MinvDistCubed;
//	ai.z = r.z * MinvDistCubed;
//}




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

	print_deviceProperties (dev, driverVersion, runtimeVersion, deviceProp);
	print_simulationParameters (nElem, nIter, NUM_CPU_THREADS);

	////////////////////////////////////////////////////////////////
	/// INITIALIZING SIMULATION
	////////////////////////////////////////////////////////////////

	float4 *h_r[2], *h_v, *h_a;
	float4 *d_r[2], *d_v, *d_a;
	float4 *h_dref_r, *h_dref_v, *h_dref_a;
	
	size_t nBytes = nElem * sizeof(float4);
	h_r[0] = (float4 *) malloc(nBytes);
	h_r[1] = (float4 *) malloc(nBytes);
	h_v    = (float4 *) malloc(nBytes);
	h_a    = (float4 *) malloc(nBytes);

	checkCudaErrors (cudaMalloc ((void**) &(d_r[0]), nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_r[1]), nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_v),    nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_a),    nBytes));

	// allocating page-locked memory for higher communication bandwidth during real-time vis.
	checkCudaErrors (cudaMallocHost ((void**) &h_dref_r, nBytes));
	checkCudaErrors (cudaMallocHost ((void**) &h_dref_v, nBytes));
	checkCudaErrors (cudaMallocHost ((void**) &h_dref_a, nBytes));

	memset (h_r[0], 0, nBytes);
	memset (h_r[1], 0, nBytes);
	memset (h_v,    0, nBytes);
	memset (h_a,    0, nBytes);


	// initialize data on host size and then transfer to device
	

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

	dim3 block_size (1024);
	dim3 grid_size	((nElem + block_size.x-1)/(block_size.x));
	unsigned int nTiles = (nElem + block_size.x-1)/block_size.x;

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
		cudaMemcpy(gref_m, d_m, nBytes, cudaMemcpyDeviceToHost);
		// cudaMemcpy(gref_r, d_r2, nBytes*2, cudaMemcpyDeviceToHost);
		// cudaMemcpy(gref_v, d_v2, nBytes*2, cudaMemcpyDeviceToHost);
		// cudaMemcpy(gref_a, d_a2, nBytes*2, cudaMemcpyDeviceToHost);

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
	
	cudaFreeHost (gref_r);
	cudaFreeHost (gref_v);
	cudaFreeHost (gref_a);

	checkCudaErrors (cudaDeviceReset());
	
	printf("Device successfully reset.\n");
	printf("\nElapsed Time: %lfs\n", elapsedTime);
	printf("Average timestep simulation duration: %lfs\n", elapsedTime/nIter); 

	free (h_m);
	free (h_r1); free (h_r2);
	free (h_v1); free (h_v2);
	free (h_a1); free (h_a2);

	return 0;
}




__device__ float3 bodyBodyInteraction (float3 ai, float4 bi, float4 bj)
{
	float3 dist;
	
	dist.x = bj.x - bi.x;
	dist.y = bj.y - bi.y;
	dist.z = bj.z - bi.z;
	
	float distSqr = dot(dist, dist) + SOFTENING;
	float invDistCube = rsqrtf(distSqr * distSqr * distSqr);
	
	float s = bj.w * invDistCube
	
	ai.x += s * r.x;
	ai.y += s * r.y;
	ai.z += s * r.z;
	return ai;
}


__device__ float4 calcAcceleration (float4 *devX, unsigned nTiles)
{
	unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float4[] shPosition4;
	
	float4 myPosition4;
	float3 acc3 = {0.0f, 0.0f, 0.0f};
	
	myPosition4 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition4[threadIdx] = devX[ tile*blockDim.x + threadIdx ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory
		#pragma unroll 4
		for (unsigned j=0, j<blockDim.x; j++)
			acc3 = bodyBodyInteraction(acc3, myPosition4, shPosition4[j]);
		
		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}
	
	float4 acc4 = {acc3.x, acc3.y, acc3.z, 0.0f};
	return acc4;
}

__global__ void calcIntegration (float4 *devX_ip1, const float *devX_i,
	float4 *devV_i, float4 *devA_i, const unsigned nElem, const unsigned nTiles)
{
	unsigned gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < nElem) {
		float4 old_acc = devA_i[gtid];
		float4 old_vel = devV_i[gtid];
		float4 old_pos = devX_i[gtid];
		
		float4 new_pos = old_pos + scalevec(old_vel, DT) + scalevec(old_acc, DTSQd2);
		float4 new_acc = calcAcceleration (devX_i, nTiles);
		float4 new_vel = old_vel + scale_vec(old_acc + new_acc, DTd2)
		
		devA_i  [gtid] = new_acc;
		devV_i  [gtid] = new_vel;
		devX_ip1[gtid] = new_pos;
	}
}


