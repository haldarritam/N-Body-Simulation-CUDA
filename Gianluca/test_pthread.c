#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>


// global constants
#define NUM_THREADS 32
#define MAX_MASS 100.0f
#define MAX_POS_X 100.0f
#define MAX_POS_Y 100.0f
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
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_condition = PTHREAD_COND_INITIALIZER;
unsigned long count = 0;


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
    //time_t t;
    //srand ((unsigned int) time(&t));
    srand ((unsigned int) 1000);

    // define local variables for convenience
    unsigned long nElem = US.nElem;

    // populating mass, position, & velocity arrays
    unsigned long idx;
    for (idx=0; idx<nElem; idx++) 
    {
        US.m[idx]     = 100.0;//(float) ((double) rand() / (double) (RAND_MAX/MAX_MASS));
        US.r1[2*idx]   = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS_X*2)) - MAX_POS_X);
        US.r1[2*idx+1] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_POS_Y*2)) - MAX_POS_Y);
        US.v1[2*idx]   = (float) ((double) rand() / (double) (RAND_MAX/(MAX_VEL_X*2)) - MAX_VEL_X);
        US.v1[2*idx+1] = (float) ((double) rand() / (double) (RAND_MAX/(MAX_VEL_Y*2)) - MAX_VEL_Y);
    }
}

void print_BodyStats (unsigned long iter)
{
    unsigned long nElem = US.nElem;
    float **r, **v, **a;

    if (iter % 2 == 0) {
        r = &(US.r1);
        v = &(US.v1);
        a = &(US.a1);
    } else {
        r = &(US.r2);
        v = &(US.v2);
        a = &(US.a2);
    }

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
            printf("%.2f\n", US.m[idx]);
        else
            printf("%.2f\t", US.m[idx]);
    }

    // print x-position
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n", *(*r + 2*idx));
        else
            printf("%.2f\t", *(*r + 2*idx));
    }

    // print y-position
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n", *(*r + 2*idx+1));
        else
            printf("%.2f\t", *(*r + 2*idx+1));
    }

    // print x-velocity
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n", *(*v + 2*idx));
        else
            printf("%.2f\t", *(*v + 2*idx));
    }

    // print y-velocity
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n", *(*v + 2*idx+1));
        else
            printf("%.2f\t", *(*v + 2*idx+1));
    }

    // print x-acceleration
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n", *(*a + 2*idx));
        else
            printf("%.2f\t", *(*a + 2*idx));
    }

    // print y-acceleration
    for (unsigned long idx=0; idx<nElem; idx++) {
        if (idx == nElem-1)
            printf("%.2f\n\n", *(*a + 2*idx+1));
        else
            printf("%.2f\t", *(*a + 2*idx+1));
    }

}


void *init_Acceleration_SMT (void *arg)
{
    // define local variables for convenience
    unsigned long start, end, len, offset, nElem;

    nElem = US.nElem;
    offset = (unsigned long) arg;
    len = (unsigned long) US.nElem / NUM_THREADS;
    start = offset * len;
    end = start + len;

    unsigned long i, j;
    float ax_ip1, ay_ip1;
    float dx_ip1, dy_ip1, rDistSquared, invDistCubed;
    float **i_r = &(US.r1);
    float **o_a = &(US.a1);

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
                dx_ip1 = *(*i_r + (2*j))   - *(*i_r + (2*i));
                dy_ip1 = *(*i_r + (2*j+1)) - *(*i_r + (2*i+1));
                rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + SOFTENING;
                invDistCubed = G*US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
                ax_ip1 += dx_ip1 * invDistCubed;
                ay_ip1 += dy_ip1 * invDistCubed;
            }
        }

        *(*o_a + (2*i))   = ax_ip1;
        *(*o_a + (2*i+1)) = ay_ip1;
    }

    pthread_exit (NULL);
}


void *computeHost_SMT (void *arg)
{
	// define local variables for convenience
	unsigned long start, end, len, offset, nElem, nIter;

	nElem = US.nElem;
	nIter = US.nIter;
	offset = (unsigned long) arg;
	len = (unsigned long) nElem / NUM_THREADS;
	start = offset * len;
	end = start + len;

	unsigned long i, j;
    float ax_ip1, ay_ip1;
    float dx_ip1, dy_ip1, rDistSquared, invDistCubed;
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
	    for (i=start; i<end; i++)
	    {
	        *(*o_r + (2*i))   = *(*i_r + (2*i))   + *(*i_v + (2*i))*DT   + *(*i_a + (2*i))  *DT2;
	        *(*o_r + (2*i+1)) = *(*i_r + (2*i+1)) + *(*i_v + (2*i+1))*DT + *(*i_a + (2*i+1))*DT2;
	    }

        // position computation done
        pthread_mutex_lock (&count_mutex);
        count++;

        if (count == NUM_THREADS*(2*iter+1)) {
            pthread_cond_broadcast (&count_condition);
            //printf("Broadcasting by tid=%ld\n", offset);
        }
        else {
            do {
                pthread_cond_wait (&count_condition, &count_mutex);
                //printf("Condition Broadcast received by tid=%ld\n", offset);
            } while (count < NUM_THREADS*(2*iter+1));
        }

        pthread_mutex_unlock (&count_mutex);


	    // calculating NEXT acceleration of each body from the position of every other bodies
	    // ... and NEXT velocity of each body utilizing the next acceleration
        #pragma GCC unroll 32
	    for (i=start; i<end; i++)
	    {
	        ax_ip1 = 0.0;
	        ay_ip1 = 0.0;
	        for (j=0; j<nElem; j++)
	        {
				dx_ip1 = *(*o_r + (2*j))   - *(*o_r + (2*i));
				dy_ip1 = *(*o_r + (2*j+1)) - *(*o_r + (2*i+1));
				rDistSquared = dx_ip1*dx_ip1 + dy_ip1*dy_ip1 + SOFTENING;
				invDistCubed = G*US.m[j]/sqrtf(rDistSquared*rDistSquared*rDistSquared);
				ax_ip1 += dx_ip1 * invDistCubed;
				ay_ip1 += dy_ip1 * invDistCubed;
	        }
	        *(*o_a + (2*i))   = ax_ip1;
	        *(*o_a + (2*i+1)) = ay_ip1;

	        *(*o_v + (2*i))   = *(*i_v + (2*i))   + (*(*i_a + (2*i))  +ax_ip1)*DT/2;
	        *(*o_v + (2*i+1)) = *(*i_v + (2*i+1)) + (*(*i_a + (2*i+1))+ay_ip1)*DT/2;
	    }


		// computation done
		pthread_mutex_lock (&count_mutex);
		count++;

		if (count == NUM_THREADS*(2*iter+2)) {
			pthread_cond_broadcast (&count_condition);
			//printf("Broadcasting by tid=%ld\n", offset);
		}
		else {
			do {
				pthread_cond_wait (&count_condition, &count_mutex);
				//printf("Condition Broadcast received by tid=%ld\n", offset);
			} while (count < NUM_THREADS*(2*iter+2));
		}

		pthread_mutex_unlock (&count_mutex);

        // if (offset == 0) {
        //     printf("x: %.6f\ty: %.6f\n", **o_r, *(*o_r+1));
        // }

		// if ((offset == 0) && (iter%1000 == 0)) {
  //           print_BodyStats (iter+1);
  //       }
	}

	pthread_exit (NULL);
}


int main(int argc, char const *argv[])
{
    if (argc > 3) {
        printf("Error: Wrong number of args. Exiting.\n");
        exit(1);
    }

    long i;
    int rc;
    void *status;
    char *ptr1, *ptr2;

    unsigned long nElem = 16000;    // argv[1]: number of elements
    unsigned long nIter = 10;      // argv[2]: number of simulation time steps
    if (argc > 0)
        nElem = strtoul(argv[1], &ptr1, 10);
    if (argc > 1)
        nIter = strtoul(argv[2], &ptr2, 10);  

    printf("\n===== Simulation Parameters =====\n\n");
    printf("  Number of Bodies = %ld\n", nElem);
    printf("  Number of Time Steps = %ld\n", nIter);
    printf("  Number of CPU Threads = %d\n\n", NUM_THREADS);
    printf("=================================\n\n");

    size_t nBytes = nElem * sizeof(float);

    // allocate space on system RAM (13 x nBytes)
    float *m = (float*) malloc(nBytes);
    float *r1 = (float*) malloc(nBytes*2);	// x,y coordinates
    float *r2 = (float*) malloc(nBytes*2);	// x,y coordinates
    float *v1 = (float*) malloc(nBytes*2);	// x,y coordinates
    float *v2 = (float*) malloc(nBytes*2);	// x,y coordinates
    float *a1 = (float*) malloc(nBytes*2);	// x,y coordinates
    float *a2 = (float*) malloc(nBytes*2);	// x,y coordinates

    // initialize mass, position,and velocity vectors and filling UNIVERSE struct
    memset (m, 0, nBytes);
    memset (r1, 0, nBytes*2);
    memset (r2, 0, nBytes*2);
    memset (v1, 0, nBytes*2);
    memset (v2, 0, nBytes*2);
    memset (a1, 0, nBytes*2);
    memset (a2, 0, nBytes*2);

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
    pthread_t threads [NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init (&attr);
    pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);


    // creating the threads
    for (i=0; i<NUM_THREADS; i++) {
        rc = pthread_create (&threads[i], &attr, init_Acceleration_SMT, (void *) i);
        if (rc) {
            printf("Error; return code from pthread_create() is %d.\n", rc);
            exit(EXIT_FAILURE);
        }
    }

    // wait on the other threads
    for (i=0; i<NUM_THREADS; i++) {
        rc = pthread_join (threads[i], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d.\n", rc);
            exit(EXIT_FAILURE);
        }
        // printf("main(): completed join with thread #%ld having status of %ld\n",
        //  i, (long) status);
    }
    printf ("%lfs\n", getTimeStamp()-time0);
    //print_BodyStats (0);

    printf("\nBeginning CPU simulation ...\n");
	double time1 = getTimeStamp();
    // creating the threads
    for (i=0; i<NUM_THREADS; i++) {
    	rc = pthread_create (&threads[i], &attr, computeHost_SMT, (void *) i);
    	if (rc) {
    		printf("Error; return code from pthread_create() is %d.\n", rc);
    		exit(EXIT_FAILURE);
    	}
    }

    // wait on the other threads
    for (i=0; i<NUM_THREADS; i++) {
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
