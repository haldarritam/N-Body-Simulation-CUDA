#include "nbody_helper2.h"


pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_condition = PTHREAD_COND_INITIALIZER;
unsigned long count = 0;

FILE *destFile;

int main(int argc, char const *argv[])
{
	if (argc > 4) {
		printf("Error: Wrong number of arguments.\n");
		exit(EXIT_FAILURE);
	}

	unsigned int nElem = 32768;
	unsigned int nIter = 100;
	unsigned int config = 0;
	char *ptr1, *ptr2, *ptr3;

	// acquiring command line arguments, if any.
	if (argc > 1)	// no. of elements
		nElem  = (unsigned int) strtoul(argv[1], &ptr1, 10);
	if (argc > 2)	// no. of iterations
		nIter  = (unsigned int) strtoul(argv[2], &ptr2, 10);
	if (argc > 3)	// initial config of bodies
		config = (unsigned int) strtoul(argv[3], &ptr3, 10);

	destFile = fopen("./out.csv", "w");	// write-only
	if (destFile == NULL) {
		printf("Error! Could not open file.\n");
		exit(EXIT_FAILURE);
	}

	print_simulationParameters (nElem, nIter, NUM_CPU_THREADS);

	///////////////////////////////////////////////////////////////////////////////
	/// INITIALIZING SIMULATION
	///////////////////////////////////////////////////////////////////////////////

	float3 *h_r[2], *h_v, *h_a;
	size_t nBytes = nElem * sizeof(float3);

	// allocate space on system RAM (4 x nBytes)
	h_r[0] = (float3*) malloc(nBytes);
	h_r[1] = (float3*) malloc(nBytes);
	h_v    = (float3*) malloc(nBytes);
	h_a    = (float3*) malloc(nBytes);

	// initialize mass, position,and velocity vectors and filling UNIVERSE struct
	printf("Initializing bodies' positions / velocities on HOST. Time taken: ");
	double time0 = getTimeStamp();
	init_MassPositionVelocity(h_r[0], h_v, nElem, config);
	printf ("%lfs\n", getTimeStamp()-time0);

	UNIVERSE US;
	US.r = h_r;
	US.v = h_v;
	US.a = h_a;
	US.nElem = nElem;
	US.nIter = nIter;

	// initialize mutex and condition variable objects
	pthread_mutex_init (&count_mutex, NULL);
	pthread_cond_init (&count_condition, NULL);

	// for portability, explicity create threads in a joinable state
	pthread_t threads [NUM_CPU_THREADS];
	pthread_attr_t attr;
	pthread_attr_init (&attr);
	pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);

	// creating the threads
	unsigned int i, rc;
	THREAD_STRUCT thd;
	for (i=0; i<NUM_CPU_THREADS; i++) {
		thd.tid = i;
		thd.system = &US;
		rc = pthread_create (&threads[i], &attr, init_Acceleration_SMT, (void *) thd);
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
		thd.tid = i;
		thd.system = &US;
		rc = pthread_create (&threads[i], &attr, computeHost_SMT, (void *) thd);
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

	free (h_r[0]); free (h_r[1]);
	free (h_v);
	free (h_a);

	pthread_attr_destroy (&attr);
	pthread_mutex_destroy (&count_mutex);
	pthread_cond_destroy (&count_condition);

	printf("\nElapsed Time: %lfs\n", elapsedTime);
	printf("Average timestep simulation duration: %lfs\n", elapsedTime/nIter); 

	pthread_exit (NULL);

	return 0;
}




