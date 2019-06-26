#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <vector>
#include <stdbool.h>
#include <SFML/Graphics.hpp>
#include "nbody_helper2.h"

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
		config = (unsigned int) strtoul(argv[3], &ptr3, 10);
	if (argc > 3) {	// no. of iterations
		nIter  = (unsigned int) strtoul(argv[2], &ptr2, 10);
		limit_iter = true;
	}


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
	/// Initializing the animation window
	/////////////////////////////////////////////////////////////////////////////////////////////////

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
		
		window.clear();
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
