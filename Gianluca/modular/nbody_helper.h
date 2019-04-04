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
	float *m;
	float *r1, *r2;
	float *v1, *v2;
	float *a1, *a2;
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

