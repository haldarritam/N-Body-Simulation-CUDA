#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <vector>
#include <SFML/Graphics.hpp>
#include "aux_functions_gpu.h"

#define CONFIG	1	

#if   CONFIG == 0
#include "aux_declarations_gpu_rtsim_1.h"
#elif CONFIG == 1
#include "aux_declarations_gpu_rtsim_2.h"
#elif CONFIG == 2
#include "aux_declarations_gpu_rtsim_3.h"
#else
#include "aux_declarations_gpu_rtsim_1.h"
#endif

std::vector <sf::CircleShape> body_graphics;

enum nBody_config
{
	NBODY_CONFIG_RANDOM,
	NBODY_CONFIG_SPIRAL,
	NBODY_CONFIG_EXPAND,
	NBODY_NUM_CONFIGS
};

// main function
int main (const int argc, const char** argv) {
    printf("\n");
    int nBodies = 30000;
    float dt = DT; // time step
    int nIters = 1000, limit_iter = 0;  // simulation iterations
    int iter = 0, i = 0, grid_size = 0, stop = 0;
    double total_time_gpu = 0;

    float* x = NULL;
    float* y = NULL;
    
    body_pos *h_body_pos, *d_body_pos;
    body_parameters *h_body_par, *d_body_par;
    
    // reading the arguments (argv data)
    switch(argc)
    {
      case 1:
        printf("------------------------------------------------------\n\n");
        printf("Default values considered, nBodies: 30000.\n\n");
        printf("------------------------------------------------------\n\n");
      break;
      case 2:
        nBodies = atoi(argv[1]);
        printf("------------------------------------------------------\n\n");
        printf("Values considered, nBodies: %i.\n\n", nBodies);
        printf("------------------------------------------------------\n\n");
      break;
      case 3:
        nBodies = atoi(argv[1]);
        nIters = atoi(argv[2]);
        limit_iter = 1;
        printf("------------------------------------------------------\n\n");
        printf("Values considered, nBodies: %i, nIters: %i.\n\n", nBodies, nIters);
        printf("------------------------------------------------------\n\n");
      break;
      default:
        printf("ERR: Invalid number of arguments passed.\n"
               "Aborting...\n");
        return -1;
    }

    // initialising the animation window
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    sf::RenderWindow window(sf::VideoMode(X_RES, Y_RES), "N-Body Simulation", sf::Style::Default, settings);
  
    // host side memory allocation
    size_t pos_bytes = nBodies*sizeof(body_pos);
    size_t par_bytes = nBodies*sizeof(body_parameters);
    cudaMallocHost((body_pos **) &h_body_pos, pos_bytes);
    cudaMallocHost((body_parameters **) &h_body_par, par_bytes);
    x = new float[nBodies];
    y = new float[nBodies];

    
    // Init mass / pos / vel / acc data
    initialize_bodies(h_body_pos, h_body_par, nBodies);             
    
    for (i = 0; i < nBodies; i++) {
      x[i] = h_body_pos[i].x;
      y[i] = h_body_pos[i].y;
    }
    // Device side memory allocation  
    cudaMalloc((body_pos **) &d_body_pos, pos_bytes);
    cudaMalloc((body_parameters **) &d_body_par, par_bytes); 
  
    // determining the grid size
    grid_size = (nBodies+BLOCK_SIZE-1)/BLOCK_SIZE;
  
    // initializing the dim3 variables  
    dim3 block( BLOCK_SIZE, 1, 1 ) ; 
    dim3 grid( grid_size, 1, 1);
    
    // starting the iterations
    printf("---------GPU Data---------\n");

    // memcopy (host -> device)
    cudaMemcpy(d_body_pos, h_body_pos, pos_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_body_par, h_body_par, par_bytes, cudaMemcpyHostToDevice);

    while((window.isOpen()  && !stop)  || 
         ((limit_iter == 1) && (iter < nIters))) {
        
        // print statements
        if(iter%(nIters/3)==0) {
            printf("iter:%i\n",iter);
            printf("MASS 0\t\t\tMASS 1\t\t\tMASS 2\n");
            printf("x:%.04f\t\tx:%.04f\t\tx:%.04f\n",h_body_pos[0].x,h_body_pos[1].x,h_body_pos[2].x);
            printf("y:%.04f\t\ty:%.04f\t\ty:%.04f\n",h_body_pos[0].y,h_body_pos[1].y,h_body_pos[2].y);
            printf("z:%.04f\t\tz:%.04f\t\tz:%.04f\n",h_body_pos[0].z,h_body_pos[1].z,h_body_pos[2].z);
            printf("\n");
        }

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        double timeStampA = getTimeStamp();
        
        // kernel call
        nbody_acc_vel<<<grid, block>>>(d_body_pos,d_body_par,dt,nBodies);
        cudaDeviceSynchronize();

        nbody_integration<<<grid, block>>>(d_body_pos,d_body_par,dt,nBodies);
        cudaDeviceSynchronize();

        // memcopy (device -> host)
        cudaMemcpy(h_body_pos, d_body_pos, pos_bytes, cudaMemcpyDeviceToHost);

        double timeStampB = getTimeStamp();
        gpuErrchk(cudaPeekAtLastError());
        
        for (i = 0; i < nBodies; i++){
            body_graphics[i].move(h_body_pos[i].x - x[i] , h_body_pos[i].y - y[i]);
            x[i] = h_body_pos[i].x;
            y[i] = h_body_pos[i].y;
        }

        window.clear();

        for (i = 0; i < nBodies; i++)
            window.draw(body_graphics[i]);
        window.display();

        total_time_gpu = total_time_gpu + (timeStampB - timeStampA);
        iter++;

        if ((limit_iter == 1) && (iter == nIters)) {
            stop = 1;
            window.close();
        }
    }

    printf("\n");
    printf("GPU -- Total Time Taken: %lf\n\n", total_time_gpu);
  
    // free memory
    delete [] x;
    delete [] y;
    cudaFreeHost(h_body_pos);
    cudaFreeHost(h_body_par);
    cudaFree(d_body_pos);
    cudaFree(d_body_par); 
    cudaDeviceReset();
    return 0;
}
  

void initialize_bodies(body_pos *b_pos, body_parameters *b_par, int n) {
	int i = 0;
	srand(1000);
	// setting up the base body shape
    sf::CircleShape shape_green(SIZE_OF_BODIES);
    shape_green.setFillColor(sf::Color::Green);
    
    sf::CircleShape shape_red(SIZE_OF_BODIES);
    shape_red.setFillColor(sf::Color::Red);

	for (i = 0; i < n; i++) {
		b_par[i].ax = 0.0f; 
		b_par[i].ay = 0.0f;
		b_par[i].vx = 0.0f;
		b_par[i].vy = 0.0f;
	}

	switch(CONFIG)
	{

		case(NBODY_CONFIG_RANDOM):
			for (i = 0; i < n; i++) {
				if (i%50 == 0)
					b_par[i].m = 10000*((rand() / (float)RAND_MAX) * MASS);
				else
					b_par[i].m = ((rand() / (float)RAND_MAX) * MASS);
		
				b_pos[i].x = ((rand() / (float)RAND_MAX) * X_RES);
				b_pos[i].y = ((rand() / (float)RAND_MAX) * Y_RES);
				
			}
			break;

		case(NBODY_CONFIG_SPIRAL):
			for (i = 0; i < n; i++) {

				b_pos[i].x = rand() % (MAX_NUMBER_X + 1 - MINIMUM_NUMBER_X) + MINIMUM_NUMBER_X;
				b_pos[i].y = rand() % (MAX_NUMBER_Y + 1 - MINIMUM_NUMBER_Y) + MINIMUM_NUMBER_Y;
		
				if (i == 0) {
					b_par[i].m = 1000000* MASS;
					b_pos[i].x = X_RES / 2;
					b_pos[i].y = Y_RES / 2;
					body_graphics.push_back(shape_red);
				} else if (i%2500 == 0) {
					b_par[i].m = ((rand() / (float)RAND_MAX) * MASS);//100000*MASS;
					body_graphics.push_back(shape_red);
				}
					
				else {
					b_par[i].m = ((rand() / (float)RAND_MAX) * MASS);
					body_graphics.push_back(shape_green);
				}
				
				body_graphics[i].setPosition(b_pos[i].x, b_pos[i].y);
				
				float dist_x = b_pos[i].x - b_pos[0].x;
				float dist_y = b_pos[i].y - b_pos[0].y;
				float dist_sqr = dist_x*dist_x + dist_y*dist_y;
				float dist = sqrtf(dist_sqr);
				
				float vel_mag = sqrtf(G*b_par[0].m/dist);
				b_par[i].vx =    dist_y/dist*vel_mag*1.2;
				b_par[i].vy = -1*dist_x/dist*vel_mag*1.2;
			}
			printf("Finished NBODY_CONFIG_SPIRAL Init\n");
			break;

		case(NBODY_CONFIG_EXPAND):
			for (i = 0; i < n; i++) {

				b_pos[i].x = rand() % (MAX_NUMBER_X + 1 - MINIMUM_NUMBER_X) + MINIMUM_NUMBER_X;
				b_pos[i].y = rand() % (MAX_NUMBER_Y + 1 - MINIMUM_NUMBER_Y) + MINIMUM_NUMBER_Y;
		
				if (i == 0) {
					b_par[i].m = 10000*((rand() / (float)RAND_MAX) * MASS);
					b_pos[i].x = X_RES / 2;
					b_pos[i].y = Y_RES / 2;
				} else if (i%150 == 0)
					b_par[i].m = 10000*MASS;//((rand() / (float)RAND_MAX) * MASS);
				else
					b_par[i].m = ((rand() / (float)RAND_MAX) * MASS);

		
				//  b_pos[i].x = ((rand() / (float)RAND_MAX) * X_RES);
				//  b_pos[i].y = ((rand() / (float)RAND_MAX) * Y_RES);
				//  b_pos[i].z = ((rand() / (float)RAND_MAX) * 500.0f);
				b_par[i].ax = 0.0f; 
				b_par[i].ay = 0.0f;
				// b_par[i].az = 0.0f; 
				b_par[i].vx = 0.0f;
				b_par[i].vy = 0.0f;
				// b_par[i].vz = 0.0f;
			}
			break;
		default:
			for (i = 0; i < n; i++) {

				b_pos[i].x = rand() % (MAX_NUMBER_X + 1 - MINIMUM_NUMBER_X) + MINIMUM_NUMBER_X;
				b_pos[i].y = rand() % (MAX_NUMBER_Y + 1 - MINIMUM_NUMBER_Y) + MINIMUM_NUMBER_Y;
		
				if (i == 0) {
					b_par[i].m = 100000*((rand() / (float)RAND_MAX) * MASS);
					b_pos[i].x = X_RES / 2;
					b_pos[i].y = Y_RES / 2;
				} else if (i%50 == 0)
					b_par[i].m = 10000*((rand() / (float)RAND_MAX) * MASS);
				else
					b_par[i].m = ((rand() / (float)RAND_MAX) * MASS);

		
				//  b_pos[i].x = ((rand() / (float)RAND_MAX) * X_RES);
				//  b_pos[i].y = ((rand() / (float)RAND_MAX) * Y_RES);
				//  b_pos[i].z = ((rand() / (float)RAND_MAX) * 500.0f);
				b_par[i].ax = 0.0f; 
				b_par[i].ay = 0.0f;
				// b_par[i].az = 0.0f; 
				b_par[i].vx = 0.0f;
				b_par[i].vy = 0.0f;
				// b_par[i].vz = 0.0f;
			}
			break;

	}
}  

__global__ void nbody_acc_vel(body_pos* b_pos, body_parameters* b_par, float dt, int n) {
  
    int idx = threadIdx.x + blockIdx.x*blockDim.x ;
	int j = 0;
	float dx = 0.0f, 
	      dy = 0.0f,
	      // dz = 0.0f,
	      sx = 0.0f, 
	      sy = 0.0f,
	      // sz = 0.0f,  
	      distSqr = 0.0f,
	      distSqr3 = 0.0f, 
	      invDist3 = 0.0f;
	
	for (j = 0; j < n; j++) {
	  dx = b_pos[j].x - b_pos[idx].x;
	  dy = b_pos[j].y - b_pos[idx].y;
	  // dz = b_pos[j].z - b_pos[idx].z;
	  distSqr = dx*dx + dy*dy /* + dz*dz */ + EPS;
	  distSqr3 = distSqr * distSqr * distSqr;      
	  invDist3 = (G * b_par[j].m)/sqrt(distSqr3);
  
	  sx += dx * invDist3; sy += dy * invDist3; 
	  // sz += dz * invDist3;
	}
  
	// acceleration calculation
	b_par[idx].ax = sx;
	b_par[idx].ay = sy;
	// b_par[idx].az += sz;
  
	// velocity calculation
	b_par[idx].vx += b_par[idx].ax * dt * DAMPING;
	b_par[idx].vy += b_par[idx].ay * dt * DAMPING;
	// b_par[idx].vz += b_par[idx].az * dt;    
}

__global__ void nbody_integration(body_pos* b_pos, body_parameters* b_par, float dt, int n) {
  
    int idx = threadIdx.x + blockIdx.x*blockDim.x ; 
	if (idx) {
		// integrate and find the new positions
		b_pos[idx].x += b_par[idx].vx*dt + b_par[idx].ax*dt*dt/2;
		b_pos[idx].y += b_par[idx].vy*dt + b_par[idx].ay*dt*dt/2;
		// b_pos[idx].z += b_par[idx].vz*dt + b_par[idx].az*dt*dt/2;
    }
}
