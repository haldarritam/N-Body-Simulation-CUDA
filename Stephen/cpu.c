#include <stdio.h>
#include <stdlib.h>
#include<sys/time.h>
#include <math.h>
#include <pthread.h>

#define MAX_INITIAL_WEIGHT 1000
#define MAX_INITIAL_RANGE 10000
#define MAX_INITIAL_VELOCITY 100
#define EPS 1e-9f
#define NUM_THREADS 32
#define BLOCK_DIM 32
#define G 100

pthread_mutex_t mutex_tid;

typedef struct float3{
    float x;
    float y;
    float z;
} float3;

typedef struct float4{
    float w;
    float x;
    float y;
    float z;
} float4;

typedef struct nbodyStruct{
    int noElems;
	  int maxIteration;
    int cur;
    float dt;
    float4 *X;
    float3 *A;
    float3 *V;
} nbodyStruct;

// time stamp function in seconds
double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;
}

//initialize velocity and position data
void initData(float4 *A, float3 *V, int noElems){
	for (int i = 0; i < noElems; i++){
		A[i].w = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_WEIGHT);
		A[i].x = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);
		A[i].y = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);
		A[i].z = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);

		V[i].x = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
		V[i].y = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
		V[i].z = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
	}
}

// host-side acceleration compution
void* h_acce(void *arg){
	nbodyStruct *n = (nbodyStruct *) arg;
	float4 *X = n -> X;
	float3 *A = n -> A;
	int noElems = n -> noElems;
	int len = noElems / NUM_THREADS + 1;

	pthread_mutex_lock(&mutex_tid);
    int start = n->cur * len;
	int end = start + len < noElems? start + len : noElems;
    n->cur++;
    pthread_mutex_unlock(&mutex_tid);

	float3 r;
	for (int i = start; i < end; i++)
	{
		A[i].x = 0.0f;
		A[i].y = 0.0f;
		A[i].z = 0.0f;
	    for (int j = 0; j < noElems; j++)
		{
        	r.x = X[j].x - X[i].x;
            r.y = X[j].y - X[i].y;
            r.z = X[j].z - X[i].z;

		    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;
            float distSixth = distSqr * distSqr * distSqr;
            float invDistCube = 1.0f/sqrtf(distSixth);
            float s = X[j].w * invDistCube * G;

            A[i].x += r.x * s;
            A[i].y += r.y * s;
            A[i].z += r.z * s;
	    }
	}
	return NULL;
}

// host-side preprocess the data
void h_preprocess(nbodyStruct *nbody, pthread_t *threads){
	float3 *V = nbody -> V;
	float3 *A = nbody -> A;
	int noElems = nbody -> noElems;
	float dt = nbody -> dt;

	nbody->cur = 0;
	for (int i = 0; i < NUM_THREADS; i++)
      pthread_create(&threads[i], NULL, h_acce, nbody);
    for (int i = 0; i < NUM_THREADS; i++)
      pthread_join(threads[i], NULL);

	for (int i = 0; i < noElems; i++){
		V[i].x += 0.5 * A[i].x * dt;
		V[i].y += 0.5 * A[i].y * dt;
		V[i].z += 0.5 * A[i].z * dt;
	}
}

void* h_updatePosition(void *arg){
	nbodyStruct *n = (nbodyStruct *) arg;
	float4 *X = n -> X;
	float3 *V = n -> V;
	int noElems = n -> noElems;
	float dt = n -> dt;
	int len = (noElems - 1) / NUM_THREADS + 1;

	pthread_mutex_lock(&mutex_tid);
    int start = n->cur * len;
	int end = start + len < noElems? start + len : noElems;
    n->cur++;
    pthread_mutex_unlock(&mutex_tid);

	for (int i = start; i < end; i++)
	{
		X[i].x += V[i].x * dt;
		X[i].y += V[i].y * dt;
		X[i].z += V[i].z * dt;
	}
	return NULL;
}

void* h_updateVelocity(void *arg){
	nbodyStruct *n = (nbodyStruct *) arg;
	float3 *A = n -> A;
	float3 *V = n -> V;
	int noElems = n -> noElems;
	float dt = n -> dt;
	int len = noElems / NUM_THREADS + 1;

	  pthread_mutex_lock(&mutex_tid);
        int start = n->cur * len;
	      int end = start + len < noElems? start + len : noElems;
        n->cur++;
    pthread_mutex_unlock(&mutex_tid);

	for (int i = start; i < end; i++)
	{
		V[i].x += A[i].x * dt;
		V[i].y += A[i].y * dt;
		V[i].z += A[i].z * dt;
	}
	return NULL;
}

// host-side integration
void h_inte(nbodyStruct *nbody, pthread_t *threads){
	nbody->cur = 0;
	for (int i = 0; i < NUM_THREADS; i++)
      pthread_create(&threads[i], NULL, h_updatePosition, nbody);
    for (int i = 0; i < NUM_THREADS; i++)
      pthread_join(threads[i], NULL);

	nbody->cur = 0;
	for (int i = 0; i < NUM_THREADS; i++)
      pthread_create(&threads[i], NULL, h_acce, nbody);
    for (int i = 0; i < NUM_THREADS; i++)
      pthread_join(threads[i], NULL);

	nbody->cur = 0;
	for (int i = 0; i < NUM_THREADS; i++)
      pthread_create(&threads[i], NULL, h_updateVelocity, nbody);
    for (int i = 0; i < NUM_THREADS; i++)
      pthread_join(threads[i], NULL);
}

// host-side function
void h_func(nbodyStruct *nbody){
	//float4 *X = nbody -> X;
	//int noElems = nbody -> noElems;
	int maxIteration = nbody -> maxIteration;

	double timeStampA = getTimeStamp();

	pthread_t *threads = (pthread_t*) malloc(NUM_THREADS * sizeof(pthread_t));

	h_preprocess(nbody, threads);
	int i = 0;

	FILE *f = fopen("position.txt", "w");
	while (i < maxIteration){
		  i++;
		  h_inte(nbody, threads);

/*      for (int j = 0; j < noElems; j++)*/
/*          fprintf(f, "%.6f %.6f %.6f\n", X[j].x, X[j].y, X[j].z);*/
	}
	fclose(f);

	double timeStampB = getTimeStamp() ;
	printf("%.6f\n", timeStampB - timeStampA);
}

int main( int argc, char *argv[] ) {

    // get program arguments
    if( argc < 2) {
        printf( "Error: wrong number of args\n" ) ;
        exit(0) ;
    }
    int noElems = atoi(argv[1]);
	float dt = 0.01;
	if (argc > 2) dt = atof(argv[2]);
	int maxIteration = 10;
	if (argc > 3) maxIteration = atoi(argv[3]);

    // alloc memory host-side
	  float4 *h_X = (float4 *) malloc( noElems * sizeof(float4) );
    float3 *h_A = (float3 *) malloc( noElems * sizeof(float3) );
	  float3 *h_V = (float3 *) malloc( noElems * sizeof(float3) );
	  nbodyStruct *h_nbody = (nbodyStruct*) malloc(sizeof(nbodyStruct));
	  h_nbody->noElems = noElems;
	  h_nbody->maxIteration = maxIteration;
	  h_nbody->dt = dt;
	  h_nbody->X = h_X;
	  h_nbody->A = h_A;
	  h_nbody->V = h_V;


    // init matrices with random data
    initData(h_X, h_V, noElems) ;
	
	double time1 = getTimeStamp();
    h_func(h_nbody);
    double time2 = getTimeStamp();
    double elapsedTime = time2-time1;
    printf("Elapsed Time: %.4lfs\n", elapsedTime);
    printf("Elapsed Time per Iteration: %.4lfs\n", elapsedTime/maxIteration);
    
    return 0;
}





