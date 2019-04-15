#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <SFML/Graphics.hpp>

// global constants
#define NUM_CPU_THREADS 32
#define ND 2				// number of spatial dimensions
#define MAX_MASS 100.0f
#define MIN_MASS 10.0f
#define MAX_POS 10000.0f
#define MAX_VEL 8192.0f
#define G 16384
#define DT 0.0019531255f
#define DTd2 0.0009765625f
#define DTSQd2 0.00000190734f
#define DAMPENING 1.0f
#define SOFTENING 1.0f
#define SIZE_OF_BODIES        0.9f     // set between -> (0.5 - 3)
#define X_RES                 1920.0f
#define Y_RES                 1080.0f
#define MAX_PIXEL             3
#define MIN_PIXEL             0.5

std::vector <sf::CircleShape> body_graphics;

enum INIT_CONFIG {
	RANDOM_SQUARE_NO_VEL,
	RANDOM_CIRCLE_NO_VEL,
	EXPAND_SHELL,
	SPIRAL_SINGLE_GALAXY,
	SPIRAL_DOUBLE_GALAXY,
	SPIRAL_QUAD_GALAXY,
	NUM_INIT_CONFIGS
};

typedef struct {
	float *m;
	float *r1, *r2;
	float *v1, *v2;
	float *a1, *a2;
	unsigned long nElem, nIter;
} UNIVERSE;

UNIVERSE US;

typedef struct float2 { 
  float x, y; 
} float2;

// time stamp function in seconds 
double getTimeStamp()
{     
    struct timeval tv;
	gettimeofday (&tv, NULL);
	return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

float rand_sign ()
{
	return (rand()-RAND_MAX) >= 0 ? 1.0 : -1.0;
}



inline void GetFrameRate(char* char_buffer, sf::Clock* clock)
{
	sf::Time time = clock->getElapsedTime();	
	sprintf(char_buffer,"Time per frame: %i ms\n", time.asMilliseconds());	
	clock->restart();
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

void init_MassPositionVelocity (const unsigned int config)
{
	// generate different seed for pseudo-random number generator
	// time_t t;
	// srand ((unsigned int) time(&t));
	srand ((unsigned int) 1000);
	// setting up the base body shape
    sf::CircleShape shape_green(SIZE_OF_BODIES);
    shape_green.setFillColor(sf::Color::Green);
    
    sf::CircleShape shape_red(SIZE_OF_BODIES);
    shape_red.setFillColor(sf::Color::Red);

	// populating mass, position, & velocity arrays
	unsigned long idx;
	float mass_range = MAX_MASS - MIN_MASS;
	float x_width = 300.0;
	float y_width = 300.0;
	float x_mid = X_RES/2;
	float x_max = (X_RES + x_width)/2;
	float x_min = (X_RES - x_width)/2;
	float y_mid = Y_RES/2;
	float y_max = (Y_RES + y_width)/2;
	float y_min = (Y_RES - y_width)/2;

	float x, y, radius, angle, system_mass, speed_factor, tangential_speed;
	float shell_radius, shell_thickness, radial_velocity;
	float2 CoM, dist, unit_dist;
	


	switch (config) {
		case RANDOM_SQUARE_NO_VEL:
			printf("Initializing positions and mass\n");
			for (idx=0; idx<US.nElem; idx++) {
				US.r1[idx*ND] = (float) ((double) rand()/RAND_MAX) * x_width + x_min;
				US.r1[idx*ND+1] = (float) ((double) rand()/RAND_MAX) * y_width + y_min;
				US.m[idx] = (float) ((double) rand()/RAND_MAX) * mass_range + MIN_MASS;
				US.v1[idx*ND]   = 0.0;
				US.v1[idx*ND+1]   = 0.0;
				// printf("Body %ld\t x: %.6f\ty: %.6f\t m: %.6f\n",
				// 	idx, r[idx].x, r[idx].y, r[idx].z);
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
			}
			break;

		case RANDOM_CIRCLE_NO_VEL:
			for (idx=0; idx<US.nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * y_width/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				US.r1[idx*ND] = x_mid + x;
				US.r1[idx*ND+1] = y_mid + y;
				US.m[idx] = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				US.v1[idx*ND]   = 0.0;
				US.v1[idx*ND+1]  = 0.0;
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
			}
			break;

		case EXPAND_SHELL:
			shell_radius = y_width/2;
			shell_thickness = 0.25*shell_radius;
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			speed_factor=0.1;

			for (idx=0; idx<US.nElem; idx++) {
				// radius is the distance of point from center of window
				radius = (float) (rand()/RAND_MAX)*shell_thickness - shell_thickness/2 + shell_radius;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				US.r1[idx*ND] = x_mid + x;
				US.r1[idx*ND+1] = y_mid + y;
				US.m[idx] = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				CoM.x += US.r1[idx*ND] * US.m[idx];
				CoM.y += US.r1[idx*ND+1] * US.m[idx];
				system_mass += US.m[idx];
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
			}
			

			CoM.x /= system_mass;
			CoM.y /= system_mass;
			//body_graphics.push_back(shape_red);
			//body_graphics[i].setPosition(CoM.x, CoM.y);
			
			for (idx=0; idx<US.nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x =  US.r1[idx*ND] - CoM.x;
				dist.y =  US.r1[idx*ND+1] - CoM.y;
				angle = (float) atan(dist.y/dist.x);
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				radial_velocity = speed_factor * sqrtf(2*G*system_mass/radius);
				US.v1[idx*ND] = radial_velocity * (float) cos(angle);
				US.v1[idx*ND+1] = radial_velocity * (float) sin(angle);
				
			}
			break;

		case SPIRAL_SINGLE_GALAXY:
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<US.nElem; idx++) {
				if (idx == 0) {
					US.r1[ND*idx] = x_mid;
					US.r1[ND*idx+1] = y_mid;
					US.m[idx] = ((float) ((double)rand()/RAND_MAX) * mass_range + MIN_MASS) * 10000;
					body_graphics.push_back(shape_red);
				} else {
					US.r1[ND*idx] = (float) ((double)rand()/RAND_MAX) * x_width + x_min;
					US.r1[ND*idx+1] = (float) ((double)rand()/RAND_MAX) * y_width + y_min;
					US.m[idx] = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
					body_graphics.push_back(shape_green);
					
				}
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
				CoM.x += US.m[idx] * US.r1[ND*idx];
				CoM.y += US.m[idx] * US.r1[ND*idx+1];
				system_mass += US.m[idx];
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<US.nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x =  US.r1[idx*ND] - CoM.x;
				dist.y =  US.r1[idx*ND+1] - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;
				
				US.v1[idx*ND] =    unit_dist.y * tangential_speed;
				US.v1[idx*ND+1] = -1*unit_dist.x * tangential_speed;
				
			}
			break;

		case SPIRAL_DOUBLE_GALAXY:
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<US.nElem; idx++) {
				if (idx == 0) {
					US.r1[ND*idx] = x_mid;
					US.r1[ND*idx+1] = y_mid;
					US.m[idx] = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
					body_graphics.push_back(shape_red);
				} else {
					US.r1[ND*idx] = (float) (rand()/RAND_MAX) * x_width + x_min;
					US.r1[ND*idx+1] = (float) (rand()/RAND_MAX) * y_width + y_min;
					US.m[idx] = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
					body_graphics.push_back(shape_green);
					
				}
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
				CoM.x += US.m[idx] * US.r1[ND*idx];
				CoM.y += US.m[idx] * US.r1[ND*idx+1];
				system_mass += US.m[idx];
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<US.nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x =  US.r1[idx*ND] - CoM.x;
				dist.y =  US.r1[idx*ND+1] - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;
				
				US.v1[idx*ND] =    unit_dist.y * tangential_speed;
				US.v1[idx*ND+1] = -1*unit_dist.x * tangential_speed;
				
			}
			break;

		case SPIRAL_QUAD_GALAXY:
			CoM = (float2) {0.0f, 0.0f};
			system_mass = 0.0;
			for (idx=0; idx<US.nElem; idx++) {
				if (idx == 0) {
					US.r1[ND*idx] = x_mid;
					US.r1[ND*idx+1] = y_mid;
					US.m[idx] = ((float) (rand()/RAND_MAX) * mass_range + MIN_MASS)*10000;
					body_graphics.push_back(shape_red);
				} else {
					US.r1[ND*idx] = (float) (rand()/RAND_MAX) * x_width + x_min;
					US.r1[ND*idx+1] = (float) (rand()/RAND_MAX) * y_width + y_min;
					US.m[idx] = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
					body_graphics.push_back(shape_green);
					
				}
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
				CoM.x += US.m[idx] * US.r1[ND*idx];
				CoM.y += US.m[idx] * US.r1[ND*idx+1];
				system_mass += US.m[idx];
			}

			CoM.x /= system_mass;
			CoM.y /= system_mass;

			for (idx=0; idx<US.nElem; idx++) {
				// radius is now the distance of point from Center of Mass (CoM)
				dist.x =  US.r1[idx*ND] - CoM.x;
				dist.y =  US.r1[idx*ND+1] - CoM.y;
				radius = sqrtf(dist.x*dist.x + dist.y*dist.y);
				unit_dist.x = dist.x / radius;
				unit_dist.y = dist.y / radius;
				tangential_speed = sqrtf(G*system_mass/radius) * 1.1;
				
				US.v1[idx*ND] =    unit_dist.y * tangential_speed;
				US.v1[idx*ND+1] = -1*unit_dist.x * tangential_speed;
				
			}
			break;

		default:
			for (idx=0; idx<US.nElem; idx++) {
				radius = (float) (rand()/RAND_MAX) * y_width/2;
				x = (float) (rand()/RAND_MAX) * radius * rand_sign();
				y = sqrt(radius*radius - x*x) * rand_sign();
				US.r1[ND*idx] = x_mid + x;
				US.r1[ND*idx+1] = y_mid + y;;
				US.m[idx] = (float) (rand()/RAND_MAX) * mass_range + MIN_MASS;
				US.v1[ND*idx] =  0.0;
				US.v1[ND*idx+1] =  0.0;
				body_graphics.push_back(shape_green);
				body_graphics[idx].setPosition(US.r1[idx*ND], US.r1[idx*ND+1]);
			}
			break;
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
