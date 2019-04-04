#include "nbody_helper.h"


pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_condition = PTHREAD_COND_INITIALIZER;
unsigned long count = 0;

FILE *destFile;


void *computeHost_SMT (void *arg);


int main(int argc, char const *argv[])
{
	if (argc > 3) {
		printf("Error: Wrong number of args. Exiting.\n");
		exit(1);
	}

	destFile = fopen("./out.csv", "w");	// write-only
	if (destFile == NULL) {
		printf("Error! Could not open file.\n");
		exit(EXIT_FAILURE);
	}

	long i;
	int rc;
	void *status;
	char *ptr1, *ptr2;

	unsigned long nElem = 16384;    // argv[1]: number of elements
	unsigned long nIter = 10;      // argv[2]: number of simulation time steps
	if (argc > 1)
		nElem = strtoul(argv[1], &ptr1, 10);
	if (argc > 2)
		nIter = strtoul(argv[2], &ptr2, 10);  

	printf("\n===== Simulation Parameters =====\n\n");
	printf("  Number of Bodies = %ld\n", nElem);
	printf("  Number of Time Steps = %ld\n", nIter);
	printf("  Number of CPU Threads = %d\n\n", NUM_CPU_THREADS);
	printf("=================================\n\n");

	size_t nBytes = nElem * sizeof(float);

	// allocate space on system RAM (13 x nBytes)
	float *m = (float*) malloc(nBytes);
	float *r1 = (float*) malloc(nBytes*ND);	// x,y coordinates
	float *r2 = (float*) malloc(nBytes*ND);	// x,y coordinates
	float *v1 = (float*) malloc(nBytes*ND);	// x,y coordinates
	float *v2 = (float*) malloc(nBytes*ND);	// x,y coordinates
	float *a1 = (float*) malloc(nBytes*ND);	// x,y coordinates
	float *a2 = (float*) malloc(nBytes*ND);	// x,y coordinates

	// initialize mass, position,and velocity vectors and filling UNIVERSE struct
	memset (m, 0, nBytes);
	memset (r1, 0, nBytes*ND);
	memset (r2, 0, nBytes*ND);
	memset (v1, 0, nBytes*ND);
	memset (v2, 0, nBytes*ND);
	memset (a1, 0, nBytes*ND);
	memset (a2, 0, nBytes*ND);

	US.m = m;
	US.r1 = r1;
	US.r2 = r2;
	US.v1 = v1;
	US.v2 = v2;
	US.a1 = a1;
	US.a2 = a2;
	US.nElem = nElem;
	US.nIter = nIter;

	printf("Initializing bodies. Time taken: ");
	double time0 = getTimeStamp();
	init_MassPositionVelocity();

	// initialize mutex and condition variable objects
	pthread_mutex_init (&count_mutex, NULL);
	pthread_cond_init (&count_condition, NULL);

	// for portability, explicity create threads in a joinable state
	pthread_t threads [NUM_CPU_THREADS];
	pthread_attr_t attr;
	pthread_attr_init (&attr);
	pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);


	// creating the threads
	for (i=0; i<NUM_CPU_THREADS; i++) {
		rc = pthread_create (&threads[i], &attr, init_Acceleration_SMT, (void *) i);
		if (rc) {
			printf("Error; return code from pthread_create() is %d.\n", rc);
			exit(EXIT_FAILURE);
		}
	}

	// wait on the other threads
	for (i=0; i<NUM_CPU_THREADS; i++) {
		rc = pthread_join (threads[i], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d.\n", rc);
			exit(EXIT_FAILURE);
		}
		// printf("main(): completed join with thread #%ld having status of %ld\n",
		//  i, (long) status);
	}
	printf ("%lfs\n", getTimeStamp()-time0);

	printf("\nBeginning CPU simulation ...\n");
	double time1 = getTimeStamp();
	// creating the threads
	for (i=0; i<NUM_CPU_THREADS; i++) {
		rc = pthread_create (&threads[i], &attr, computeHost_SMT, (void *) i);
		if (rc) {
			printf("Error; return code from pthread_create() is %d.\n", rc);
			exit(EXIT_FAILURE);
		}
	}

	// wait on the other threads
	for (i=0; i<NUM_CPU_THREADS; i++) {
		rc = pthread_join (threads[i], &status);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d.\n", rc);
			exit(EXIT_FAILURE);
		}
		// printf("main(): completed join with thread #%ld having status of %ld\n",
		// 	i, (long) status);
	}

	double time2 = getTimeStamp();
	double elapsedTime = time2-time1;

	fclose(destFile);

	free (m);
	free (r1); free (r2);
	free (v1); free (v2);
	free (a1); free (a2);

	pthread_attr_destroy (&attr);
	pthread_mutex_destroy (&count_mutex);
	pthread_cond_destroy (&count_condition);

	printf("\nElapsed Time: %lfs\n", elapsedTime);
	printf("Average timestep simulation duration: %lfs\n", elapsedTime/nIter); 

	pthread_exit (NULL);

	return 0;
}


void *computeHost_SMT (void *arg)
{
	// define local variables for convenience
	unsigned long start, end, len, offset, nElem, nIter;

	nElem = US.nElem;
	nIter = US.nIter;
	offset = (unsigned long) arg;
	len = (unsigned long) nElem / NUM_CPU_THREADS;
	start = offset * len;
	end = start + len;

	unsigned long i, j;
	float ax_ip1, ay_ip1, az_ip1;
	float dx_ip1, dy_ip1, dz_ip1, rDistSquared, MinvDistCubed;
	float **i_r, **i_v, **i_a;
	float **o_r, **o_v, **o_a;
	for (unsigned long iter=0; iter<nIter; iter++) {

		// since the computation cannot be done inplace, we constantly need to 
		// swap where the input and output data locations are
		if (iter % 2 == 0) {
			i_r = &(US.r1);
			i_v = &(US.v1);
			i_a = &(US.a1);
			o_r = &(US.r2);
			o_v = &(US.v2);
			o_a = &(US.a2);
		} else {
			i_r = &(US.r2);
			i_v = &(US.v2);
			i_a = &(US.a2);
			o_r = &(US.r1);
			o_v = &(US.v1);
			o_a = &(US.a1);
		}

		// calculating NEXT position of each body
		for (i=start; i<end; i++) {
			if (i % 100 == 0) {
				*(*o_r + (ND*i))   = *(*i_r + (ND*i));
				*(*o_r + (ND*i+1)) = *(*i_r + (ND*i+1));
				*(*o_r + (ND*i+2)) = *(*i_r + (ND*i+2));
			} else {
				*(*o_r + (ND*i))   = *(*i_r + (ND*i))   + *(*i_v + (ND*i))*DT   + *(*i_a + (ND*i))  *DTSQd2;
				*(*o_r + (ND*i+1)) = *(*i_r + (ND*i+1)) + *(*i_v + (ND*i+1))*DT + *(*i_a + (ND*i+1))*DTSQd2;
				*(*o_r + (ND*i+2)) = *(*i_r + (ND*i+2)) + *(*i_v + (ND*i+2))*DT + *(*i_a + (ND*i+2))*DTSQd2;
			}
		}

		// position computation done
		pthread_mutex_lock (&count_mutex);
		count++;

		if (count == NUM_CPU_THREADS*(2*iter+1)) {
			pthread_cond_broadcast (&count_condition);
			//printf("Broadcasting by tid=%ld\n", offset);
		} else {
			do {
				pthread_cond_wait (&count_condition, &count_mutex);
				//printf("Condition Broadcast received by tid=%ld\n", offset);
			} while (count < NUM_CPU_THREADS*(2*iter+1));
		}

		pthread_mutex_unlock (&count_mutex);


		// calculating NEXT acceleration of each body from the position of every other bodies
		// ... and NEXT velocity of each body utilizing the next acceleration
		for (i=start; i<end; i++) {
			ax_ip1 = 0.0;
			ay_ip1 = 0.0;
			az_ip1 = 0.0;

			// unrolling this loop 8x for ~2% performance improvement
			j = 0;
			//#pragma GCC unroll 8
			while (j < nElem) {
				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #1

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #2

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #3

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #4

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #5

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #6

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));


				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #7

				dx_ip1 = *(*o_r + (ND*j+0)) - *(*o_r + (ND*i+0));
				dy_ip1 = *(*o_r + (ND*j+1)) - *(*o_r + (ND*i+1));
				dz_ip1 = *(*o_r + (ND*j+2)) - *(*o_r + (ND*i+2));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + dz_ip1*dz_ip1 + SOFTENING;
				MinvDistCubed = US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * MinvDistCubed;
				ay_ip1 += dy_ip1 * MinvDistCubed;
				az_ip1 += dz_ip1 * MinvDistCubed;

				j++; // unroll #8
			}
			
			*(*o_a + (ND*i+0)) = G*ax_ip1;
			*(*o_a + (ND*i+1)) = G*ay_ip1;
			*(*o_a + (ND*i+2)) = G*az_ip1;

			*(*o_v + (ND*i+0)) = *(*i_v + (ND*i+0)) + (*(*i_a + (ND*i+0))+ax_ip1)*DTd2;
			*(*o_v + (ND*i+1)) = *(*i_v + (ND*i+1)) + (*(*i_a + (ND*i+1))+ay_ip1)*DTd2;
			*(*o_v + (ND*i+2)) = *(*i_v + (ND*i+2)) + (*(*i_a + (ND*i+2))+az_ip1)*DTd2;
		}


		// computation done
		pthread_mutex_lock (&count_mutex);
		count++;

		if (count == NUM_CPU_THREADS*(2*iter+2)) {
			// writing to file
			for (unsigned int idx=0; idx<nElem; idx++) {
				fprintf(destFile, "%f,%f,%f\n", *(*o_r+ND*idx+0), *(*o_r+ND*idx+1), *(*o_r+ND*idx+2));
			}

			pthread_cond_broadcast (&count_condition);
			//printf("Broadcasting by tid=%ld\n", offset);
		} else {
			do {
				pthread_cond_wait (&count_condition, &count_mutex);
				//printf("Condition Broadcast received by tid=%ld\n", offset);
			} while (count < NUM_CPU_THREADS*(2*iter+2));
		}

		pthread_mutex_unlock (&count_mutex);

		if (offset == 0) {
			printf("%ld:\tx: %.6f\ty: %.6f\tz: %.6f\n", iter, **o_r, *(*o_r+1), *(*o_r+2));
		}

		//if (offset == 0)
		//	print_BodyStats (US.m, *o_r, *o_v, *o_a);
	}

	pthread_exit (NULL);
}

