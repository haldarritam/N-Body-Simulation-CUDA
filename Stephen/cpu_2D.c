#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>

#define MAX_INITIAL_WEIGHT 100
#define MAX_INITIAL_RANGE 100
#define MAX_INITIAL_VELOCITY 1
#define EPS 0.0001f
#define NUM_THREADS 32
#define G 8

pthread_mutex_t mutex_tid;

typedef struct float2 {
        float x;
        float y;
} float2;

typedef struct float3 {
        float w;
        float x;
        float y;
} float3;

typedef struct nbodyStruct {
        int noElems;
        int maxIteration;
        int cur;
        float dt;
        float3 *X;
        float2 *A;
        float2 *V;
} nbodyStruct;

nbodyStruct *h_nbody;

// time stamp function in seconds
double getTimeStamp() {
        struct timeval tv;
        gettimeofday( &tv, NULL );
        return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

//initialize velocity and position data
void initData(float3 *A, float2 *V, int noElems){
        for (int i = 0; i < noElems; i++) {
                A[i].w = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_WEIGHT);
                A[i].x = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);
                A[i].y = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);

                V[i].x = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
                V[i].y = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
        }
}

// host-side acceleration compution
void* h_acce(void *arg){
        nbodyStruct *n = (nbodyStruct *) arg;
        float3 *X = n->X;
        float2 *A = n->A;
        int noElems = n->noElems;
        int len = noElems / NUM_THREADS + 1;

        pthread_mutex_lock(&mutex_tid);
        int start = n->cur * len;
        int end = start + len < noElems ? start + len : noElems;
        n->cur++;
        pthread_mutex_unlock(&mutex_tid);

        float2 r;
        for (int i = start; i < end; i++)
        {
                A[i].x = 0.0f;
                A[i].y = 0.0f;
                for (int j = 0; j < noElems; j++)
                {
                        r.x = X[j].x - X[i].x;
                        r.y = X[j].y - X[i].y;

                        float distSqr = r.x * r.x + r.y * r.y + EPS;
                        float s = X[j].w * G / sqrtf(distSqr * distSqr * distSqr);

                        A[i].x += r.x * s;
                        A[i].y += r.y * s;
                }
        }
        return NULL;
}

// host-side preprocess the data
void h_preprocess(nbodyStruct *nbody, pthread_t *threads){
        float2 *V = nbody->V;
        float2 *A = nbody->A;
        int noElems = nbody->noElems;
        float dt = nbody->dt;

        nbody->cur = 0;
        for (int i = 0; i < NUM_THREADS; i++)
                pthread_create(&threads[i], NULL, h_acce, nbody);
        for (int i = 0; i < NUM_THREADS; i++)
                pthread_join(threads[i], NULL);

        for (int i = 0; i < noElems; i++) {
                V[i].x += 0.5 * A[i].x * dt;
                V[i].y += 0.5 * A[i].y * dt;
        }
}

void* h_updatePosition(void *arg){
        nbodyStruct *n = (nbodyStruct *) arg;
        float3 *X = n->X;
        float2 *V = n->V;
        int noElems = n->noElems;
        float dt = n->dt;
        int len = (noElems - 1) / NUM_THREADS + 1;

        pthread_mutex_lock(&mutex_tid);
        int start = n->cur * len;
        int end = start + len < noElems ? start + len : noElems;
        n->cur++;
        pthread_mutex_unlock(&mutex_tid);

        for (int i = start; i < end; i++)
        {
                X[i].x += V[i].x * dt;
                X[i].y += V[i].y * dt;
        }
        return NULL;
}

void* h_updateVelocity(void *arg){
        nbodyStruct *n = (nbodyStruct *) arg;
        float2 *A = n->A;
        float2 *V = n->V;
        int noElems = n->noElems;
        float dt = n->dt;
        int len = noElems / NUM_THREADS + 1;

        pthread_mutex_lock(&mutex_tid);
        int start = n->cur * len;
        int end = start + len < noElems ? start + len : noElems;
        n->cur++;
        pthread_mutex_unlock(&mutex_tid);

        for (int i = start; i < end; i++)
        {
                V[i].x += A[i].x * dt;
                V[i].y += A[i].y * dt;
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
        int maxIteration = nbody->maxIteration;

        pthread_t *threads = (pthread_t*) malloc(NUM_THREADS * sizeof(pthread_t));

        h_preprocess(nbody, threads);
        int i = 0;

        FILE *f = fopen("position.txt", "w");
        while (i < maxIteration) {
                i++;
                h_inte(nbody, threads);

/*      for (int j = 0; j < noElems; j++)*/
/*          fprintf(f, "%.6f %.6f %.6f\n", X[j].x, X[j].y, X[j].z);*/
        }
        fclose(f);
}

int main( int argc, char *argv[] ) {

        // get program arguments
        if( argc < 2) {
                printf( "Error: wrong number of args\n" );
                exit(0);
        }
        int noElems = atoi(argv[1]);
        float dt = 0.01;
        if (argc > 2) dt = atof(argv[2]);
        int maxIteration = 10;
        if (argc > 3) maxIteration = atoi(argv[3]);

        // alloc memory host-side
        float3 *h_X = (float3 *) malloc( noElems * sizeof(float3) );
        float2 *h_A = (float2 *) malloc( noElems * sizeof(float2) );
        float2 *h_V = (float2 *) malloc( noElems * sizeof(float2) );
        h_nbody = (nbodyStruct*) malloc(sizeof(nbodyStruct));
        h_nbody->noElems = noElems;
        h_nbody->maxIteration = maxIteration;
        h_nbody->dt = dt;
        h_nbody->X = h_X;
        h_nbody->A = h_A;
        h_nbody->V = h_V;


        // init matrices with random data
        initData(h_X, h_V, noElems);

        double time1 = getTimeStamp();
        h_func(h_nbody);
        double time2 = getTimeStamp();
        double elapsedTime = time2-time1;
        printf("Elapsed Time: %.4lfs\n", elapsedTime);
        printf("Elapsed Time per Iteration: %.4lfs\n", elapsedTime/maxIteration);

        return 0;
}
