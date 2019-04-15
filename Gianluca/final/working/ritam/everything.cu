#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <vector>
#include <stdbool.h>
#include <SFML/Graphics.hpp>
#include "nbody_helper2.h"

std::vector<sf::CircleShape> body_graphics;


int main (int argc, char *argv[])
{
	if (argc > 4) {
		printf("Error: Wrong number of arguments.\n");
		exit(EXIT_FAILURE);
	}

	unsigned int nElem = 32768;
	unsigned int config = 0;
	unsigned int nIter = 100;
	bool limit_iter = false;
	char *ptr1, *ptr2, *ptr3;

	// acquiring command line arguments, if any.
	if (argc > 1)	// no. of elements
		nElem  = (unsigned int) strtoul(argv[1], &ptr1, 10);
	if (argc > 2)	// initial config of bodies
		config = (unsigned int) strtoul(argv[2], &ptr3, 10);
	if (argc > 3) {	// no. of iterations
		nIter  = (unsigned int) strtoul(argv[3], &ptr2, 10);
		limit_iter = true;
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// SETTING UP DEVICE
	/////////////////////////////////////////////////////////////////////////////////////////////////

	printf("Setting up device.\n");
	int dev = 0, driverVersion = 0, runtimeVersion = 0;
	cudaDeviceProp deviceProp;
	checkCudaErrors (cudaGetDeviceProperties (&deviceProp, dev));
	checkCudaErrors (cudaSetDevice (dev));
	checkCudaErrors (cudaDriverGetVersion (&driverVersion));
	checkCudaErrors (cudaRuntimeGetVersion (&runtimeVersion));

	print_deviceProperties (dev, driverVersion, runtimeVersion, deviceProp);
	print_simulationParameters (nElem, nIter, NUM_CPU_THREADS);


	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// Initializing the animation window
	/////////////////////////////////////////////////////////////////////////////////////////////////

	printf("Initializing animation window.\n");
	char char_buffer[20] = "Time per frame: 0\n";

	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;
	sf::RenderWindow window(sf::VideoMode(X_RES, Y_RES), "N-Body Simulation", sf::Style::Default, settings);

	sf::Font font;
	if(!font.loadFromFile("./font/bignoodletitling/big_noodle_titling.ttf")) {
		printf("Error while loding font.\n");
		return EXIT_FAILURE;
	}

	sf::Text text;
	text.setFont(font);
	text.setString(char_buffer);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Red);

	sf::Clock real_clock;
	sf::Clock* clock = &real_clock;

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// INITIALIZING SIMULATION
	/////////////////////////////////////////////////////////////////////////////////////////////////

	float3 *h_r[2], *h_v;
	float3 *d_r[2], *d_v, *d_a;

	size_t nBytes = nElem * sizeof(float3);

	// allocating page-locked memory fobattle for wenothr higher communication bandwidth during real-time vis.
	checkCudaErrors (cudaMallocHost ((void**) &h_r[0], nBytes));
	checkCudaErrors (cudaMallocHost ((void**) &h_r[1], nBytes));
	checkCudaErrors (cudaMallocHost ((void**) &h_v,    nBytes));

	checkCudaErrors (cudaMalloc ((void**) &(d_r[0]), nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_r[1]), nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_v),    nBytes));
	checkCudaErrors (cudaMalloc ((void**) &(d_a),    nBytes));


	printf("Initializing bodies' positions / velocities on HOST. Time taken: ");
	double time0 = getTimeStamp();
	printf("Initializing Mass Position Velocity.\n");
	init_MassPositionVelocity(h_r[0], h_v, nElem, config);
	printf ("%lfs\n", getTimeStamp()-time0);
	//print_BodyStats(h_m, h_r1, h_v1, h_a1);

	// setting shmem and L1 cache config.
	// 		cudaFuncCachePreferNone:	no preference (default)
	//		cudaFuncCachePreferShared:	prefer 48kB shared memory and 16kB L1 cache
	//		cudaFuncCachePreferL1:		prefer 48kB L1 cache and 16kB shmem
	//		cudaFuncCachePreferEqual:	prefer 32kB L1 cache and 32kB shmem
	cudaFuncCache cacheConfig = cudaFuncCachePreferShared;
	checkCudaErrors (cudaDeviceSetCacheConfig (cacheConfig));

	// copying initialized data from host to device
	checkCudaErrors (cudaMemcpy (d_r[0], h_r[0], nBytes, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_r[1], h_r[0], nBytes, cudaMemcpyHostToDevice));
	checkCudaErrors (cudaMemcpy (d_v,    h_v,    nBytes, cudaMemcpyHostToDevice));

	// compute initial acceleration of bodies on device
	dim3 block_size (BLOCK_SIZE);
	dim3 grid_size	((nElem + block_size.x-1)/(block_size.x));
	unsigned int nTiles = (nElem + block_size.x-1)/block_size.x;
	printf("Computing initial acceleration on device. Time Taken: ");
	time0 = getTimeStamp();
	printf("Init Accleration.\n");
	initAcceleration <<<grid_size, block_size, 0, 0>>> (d_a, d_r[0], nTiles);
	cudaDeviceSynchronize ();
	printf ("%lfs\n", getTimeStamp()-time0);

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// PERFORMING SIMULATION ON DEVICE
	/////////////////////////////////////////////////////////////////////////////////////////////////

	// creating 2 streams for compute and for d2h communication
	cudaStream_t *streams = (cudaStream_t *) malloc(2*sizeof(cudaStream_t));
	cudaStreamCreate (&streams[0]);	// d2h communication
	cudaStreamCreate (&streams[1]);	// compute

	printf("Computing positions on device.\n");
	double timestamp_GPU_start = getTimeStamp();

	unsigned iter=0, stop=0;
	double time10, time11;

	printf("Just Before while loop.\n");

	while ((window.isOpen() && !stop) || (limit_iter && (iter < nIter))) {
		cudaMemcpyAsync (h_r[iter%2], d_r[iter%2], nBytes, cudaMemcpyDeviceToHost, streams[0]);
		calcIntegration <<<grid_size, block_size, 0, streams[1]>>> (
			d_r[(iter+1)%2],	// pointer to new positions
			d_r[iter%2], 		// pointer to curr positions
			d_v, 				// pointer to curr velocities
			d_a, 				// pointer to curr accelerations
			nElem, 				// number of bodies in simulation
			nTiles);			// number of shared memory sections per block

		checkCudaErrors (cudaStreamSynchronize (streams[1]));

		time10 = getTimeStamp();
		window.clear();
		time11 = getTimeStamp();
		#pragma unroll(16)
		for (unsigned elem=0; elem<nElem; elem++) {
			body_graphics[elem].setPosition(h_r[iter%2][elem].x, h_r[iter%2][elem].y);
			window.draw(body_graphics[elem]);
		}

		// displaying frame time in window
		GetFrameRate(char_buffer, clock);
		text.setString(char_buffer);
		window.draw(text);
		window.display();

		iter++;

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}
		printf("%.4fms\t%0.4fms\n",
			(time11-time10)*1000, (getTimeStamp()-time11)*1000);

		if (limit_iter && (iter == nIter)) {
			stop = 1;
			window.close();
		}
	}

	double timestamp_GPU_end = getTimeStamp();
	double elapsedTime = timestamp_GPU_end - timestamp_GPU_start;
	printf("Elapsed Time (total): %.6lfs\n", elapsedTime);
	printf("Elapsed Time per Iteration: %.6lfs\n", elapsedTime/nIter);

	/////////////////////////////////////////////////////////////////////////////////////////////////
	/// SIMULATION COMPLETE -- (free memory)
	/////////////////////////////////////////////////////////////////////////////////////////////////

	printf("Freeing global and system memory.\n");

	// stream resources are being released
	checkCudaErrors (cudaStreamDestroy (streams[0]));
	checkCudaErrors (cudaStreamDestroy (streams[1]));

	// free global memory on GPU DRAM
	checkCudaErrors (cudaFree (d_r[0]));
	checkCudaErrors (cudaFree (d_r[1]));
	checkCudaErrors (cudaFree (d_v));
	checkCudaErrors (cudaFree (d_a));

	// free page-locked ("pinned") memory on system DRAM
	checkCudaErrors (cudaFreeHost (h_r[0]));
	checkCudaErrors (cudaFreeHost (h_r[1]));
	checkCudaErrors (cudaFreeHost (h_v));

	checkCudaErrors (cudaDeviceReset());
	printf("Device successfully reset.\n");

	return 0;
}



























// time stamp function in seconds
double getTimeStamp()
{
    struct timeval tv;
	gettimeofday (&tv, NULL);
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

// HELPER FUNCTIONS
inline float3 scalevec (float3 &v0, float scalar)
{
	float3 rt = v0;
	rt.x *= scalar;
	rt.y *= scalar;
	return rt;
}

inline float dot (float2 v0, float2 v1)
{
	return v0.x*v1.x + v0.y*v1.y;
}

inline float normalize (float2 v0)
{
	float dist = sqrtf(dot(v0, v0));
	if (dist > 1e-6) {
		v0.x /= dist;
		v0.y /= dist;
	} else {
		v0.x *= 1e6;
		v0.y *= 1e6;
	}

	return dist;
}

inline float3 cross (float3 v0, float3 v1)
{
	float3 v2;
	v2.x = v0.y*v1.z - v0.z*v1.y;
	v2.y = v0.z*v1.x - v0.x*v1.z;
	v2.z = v0.z*v1.y - v0.y*v1.x;
	return v2;
}



inline float rand_sign ()
{
	return (rand()-RAND_MAX/2) >= 0 ? 1.0 : -1.0;
}

inline void GetFrameRate(char* char_buffer, sf::Clock* clock)
{
	sf::Time time = clock->getElapsedTime();
	sprintf(char_buffer,"Time per frame: %i ms\n", time.asMilliseconds());
	clock->restart();
}


void init_MassPositionVelocity (float3 *r, float3 *v, const unsigned long nElem,
	const unsigned int config)
{
	// generate different seed for pseudo-random number generator
	// time_t t;
	// srand ((unsigned int) time(&t));
	srand ((unsigned int) 1000);

	// populating mass, position, & velocity arrays
	unsigned int idx;
	float mass_range = MAX_MASS - MIN_MASS;
	float x_width = 300.0;
	float y_width = 300.0;
	float x_mid = X_RES/2+1;
	float x_min = (X_RES - x_width)/2;
	float y_mid = Y_RES/2+1;
	float y_min = (Y_RES - y_width)/2;

	float x, y, radius, angle, system_mass, speed_factor, tangential_speed;
	float shell_radius, shell_thickness, radial_velocity;
	float2 CoM, dist, unit_dist;

	// graphics variables
	sf::CircleShape shape_green(SIZE_OF_BODIES);
	shape_green.setFillColor(sf::Color::Green);

	sf::CircleShape shape_red(SIZE_OF_BODIES);
	shape_red.setFillColor(sf::Color::Red);

	switch (config) {
		case RANDOM_SQUARE_NO_VEL:
			printf("Initializing positions and mass\n");
			for (idx=0; idx<nElem; idx++) {
				r[idx].x = (float) ((double) rand()/RAND_MAX) * x_width + x_min;
				r[idx].y = (float) ((double) rand()/RAND_MAX) * y_width + y_min;
				r[idx].z = (float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = (float3) {0.0f, 0.0f, 0.0f};
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(r[idx].x, r[idx].y);
				// printf("Body %ld\t x: %.6f\ty: %.6f\t m: %.6f\n",
				// 	idx, r[idx].x, r[idx].y, r[idx].z);
			}
			break;

		case RANDOM_CIRCLE_NO_VEL:
			for (idx=0; idx<nElem; idx++) {
				radius = (float) ((double) rand()/RAND_MAX) * y_width/2;
				x = (float) ((double) rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = (float3) {0.0f, 0.0f, 0.0f};
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(r[idx].x, r[idx].y);
				if (idx % 2048 == 0) {
					printf("Body: %d\n", idx);
					printf("  Radius: %.2f\n", radius);
					printf("       x: %.2f\n", x);
					printf("       y: %.2f\n\n", y);
				}
			}
			break;

		case EXPAND_SHELL:
			shell_radius = y_width/2;
			shell_thickness = 0.25*shell_radius;
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			speed_factor=0.1f;

			for (idx=0; idx<nElem; idx++) {
				// radius is the distance of point from center of window
				radius = (float) ((double) rand()/RAND_MAX)*shell_thickness - shell_thickness/2 + shell_radius;
				x = (float) ((double) rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS;
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(r[idx].x, r[idx].y);
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				angle = (float) atan(dist.y/dist.x);
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				radial_velocity = speed_factor * sqrtf(2*G*system_mass/radius);
				v[idx].x = radial_velocity * (float) cos(angle);
				v[idx].y = radial_velocity * (float) sin(angle);
				v[idx].z = 0.0f;
			}
			break;

		case SPIRAL_SINGLE_GALAXY:
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = 960;
					r[idx].y = 540;
					r[idx].z = ((float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS)*100000;
				} else {
					r[idx].x = (float) ((double) rand()/RAND_MAX) * x_width + x_min;
					r[idx].y = (float) ((double) rand()/RAND_MAX) * y_width + y_min;
					r[idx].z = (float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS;
				}
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(r[idx].x, r[idx].y);
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				if (idx == 0) {
					v[idx].x = 0.0f;
					v[idx].y = 0.0f;
					v[idx].z = 0.0f;
				} else {
					dist.x = r[idx].x - CoM.x;
					dist.y = r[idx].y - CoM.y;
					radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
					unit_dist.x = dist.x / radius;
					unit_dist.y = dist.y / radius;
					tangential_speed = sqrtf(G*system_mass/radius) * 0.8f;

					v[idx].x =    unit_dist.y * tangential_speed;
					v[idx].y = -1*unit_dist.x * tangential_speed;
					v[idx].z = 0.0f;
				}
			}
			break;

		case SPIRAL_DOUBLE_GALAXY:
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * x_width + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * y_width + y_min;
					r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				}
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;

				v[idx].x =    unit_dist.y * tangential_speed;
				v[idx].y = -1*unit_dist.x * tangential_speed;
				v[idx].z = 0.0f;
			}
			break;

		case SPIRAL_QUAD_GALAXY:
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<nElem; idx++) {
				if (idx == 0) {
					r[idx].x = x_mid;
					r[idx].y = y_mid;
					r[idx].z = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
				} else {
					r[idx].x = (float) (rand()/RAND_MAX) * x_width + x_min;
					r[idx].y = (float) (rand()/RAND_MAX) * y_width + y_min;
					r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				}
				CoM.x += r[idx].z * r[idx].x;
				CoM.y += r[idx].z * r[idx].y;
				system_mass += r[idx].z;
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x = r[idx].x - CoM.x;
				dist.y = r[idx].y - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;

				v[idx].x =    unit_dist.y * tangential_speed;
				v[idx].y = -1*unit_dist.x * tangential_speed;
				v[idx].z = 0.0f;
			}
			break;

		default:
			for (idx=0; idx<nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * y_width/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				r[idx].x = x_mid + x;
				r[idx].y = y_mid + y;;
				r[idx].z = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				v[idx]   = (float3) {0.0f, 0.0f, 0.0f};
			}
			break;
	}
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

__device__ float2 bodyBodyInteraction (float2 ai, const float3 bi, const float3 bj)
{
	float2 dist;

	dist.x = bj.x - bi.x;
	dist.y = bj.y - bi.y;

	float distSqr = dist.x*dist.x + dist.y*dist.y + SOFTENING;
	float invDistCube = rsqrtf(distSqr * distSqr * distSqr);

	float s = bj.z * invDistCube;

	ai.x += s * dist.x;
	ai.y += s * dist.y;
	return ai;
}

__global__ void initAcceleration (float3 *devA, const float3 *__restrict__ devX, const unsigned nTiles)
{
	unsigned int gtid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	__shared__ float3 shPosition3[BLOCK_SIZE];

	float3 myPosition3;
	float2 acc2 = {0.0f, 0.0f};

	myPosition3 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition3[threadIdx.x] = devX[ tile*BLOCK_SIZE + threadIdx.x ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory
		#pragma unroll (16)
		for (unsigned j=0; j<BLOCK_SIZE; j++)
			acc2 = bodyBodyInteraction(acc2, myPosition3, shPosition3[j]);

		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}

	devA[gtid] = (float3) {G*acc2.x, G*acc2.y, 0.0f};
}

__device__ float3 calcAcceleration (const float3 *__restrict__ devX, const unsigned nTiles)
{
	unsigned int gtid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	__shared__ float3 shPosition3[BLOCK_SIZE];

	float3 myPosition3;
	float2 acc2 = {0.0f, 0.0f};

	myPosition3 = devX[gtid];
	for (unsigned tile=0; tile<nTiles; tile++) {
		shPosition3[threadIdx.x] = devX[ tile*BLOCK_SIZE + threadIdx.x ];
		__syncthreads();	// Wait for all threads in block to load data
							// ... into shared memory

		#pragma unroll (16)
		for (unsigned j=0; j<BLOCK_SIZE; j++)
			acc2 = bodyBodyInteraction(acc2, myPosition3, shPosition3[j]);

		__syncthreads();	// wait for all threads in block to complete their
							// ... computations to not overwrite sh. mem.
	}

	float3 acc3 = {G*acc2.x, G*acc2.y, 0.0f};
	return acc3;
}

__global__ void calcIntegration (float3 *devX_ip1, const float3 *__restrict__ devX_i,
	float3 *devV_i, float3 *devA_i, const unsigned nElem, const unsigned nTiles)
{
	unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid < nElem) {
		// if (gtid == 1)
		// 	printf("x: %.6f\ty: %.6f\tm: %.6f\n", devX_i[gtid].x, devX_i[gtid].y, devX_i[gtid].z);
		float3 old_acc = devA_i[gtid];
		float3 old_vel = devV_i[gtid];
		float3 old_pos = devX_i[gtid];

		devX_ip1[gtid].x = old_pos.x + old_vel.x*DT + old_acc.x*DTSQd2;
		devX_ip1[gtid].y = old_pos.y + old_vel.y*DT + old_acc.y*DTSQd2;
		float3 new_acc   = calcAcceleration (devX_i, nTiles);
		devV_i  [gtid].x = old_vel.x + (old_acc.x + new_acc.x)*DTd2;
		devV_i  [gtid].y = old_vel.y + (old_acc.y + new_acc.y)*DTd2;
		devA_i  [gtid]   = new_acc;
	}
}
