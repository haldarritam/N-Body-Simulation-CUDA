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
#define DT 0.001f

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
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_condition = PTHREAD_COND_INITIALIZER;
unsigned long count = 0;

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
                        float distSixth = distSqr * distSqr * distSqr;
                        float invDistCube = 1.0f/sqrtf(distSixth);
                        float s = X[j].w * invDistCube * G;

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

        //double timeStampA = getTimeStamp();

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

        //double timeStampB = getTimeStamp();
        //printf("%.6f\n", timeStampB - timeStampA);
}

void *computeHost_SMT (void *arg)
{
        // define local variables for convenience
        unsigned long start, end, len, offset, noElems, nIter;

        noElems = h_nbody->noElems;
        nIter = h_nbody->maxIteration;
        float3 *X = h_nbody->X;
        float2 *V = h_nbody->V;
        float2 *A = h_nbody->A;
        offset = (unsigned long) arg;
        len = (unsigned long) noElems / NUM_THREADS;
        start = offset * len;
        end = start + len;

        unsigned long i, j;
        float2 a_i, r;
        float3 x_i;
        float dx_ip1, dy_ip1, rDistSquared, invDistCubed;
        for (unsigned long iter=0; iter<nIter; iter++) {

                // calculating NEXT position of each body
                for (i=start; i<end; i++)
                {
                        X[i].x += V[i].x * DT;
                        X[i].y += V[i].y * DT;
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
                for (i=start; i<end; i++)
                {
                        a_i.x = 0.0f;
                        a_i.y = 0.0f;
                        x_i.x = X[i].x;
                        x_i.y = X[i].y;


                        for (j = 0; j < noElems; j++)
                        {
                                r.x = X[j].x - x_i.x;
                                r.y = X[j].y - x_i.y;

                                float distSqr = r.x * r.x + r.y * r.y + EPS;
                                float s = X[j].w * G / sqrtf(distSqr * distSqr * distSqr);

                                a_i.x += r.x * s;
                                a_i.y += r.y * s;
                        }

                        A[i].x = a_i.x;
                        A[i].y = a_i.y;

                        V[i].x += A[i].x * DT;
                        V[i].y += A[i].y * DT;
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

        }

        pthread_exit (NULL);
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

        int rc;
        void *status;
        // initialize mutex and condition variable objects
        pthread_mutex_init (&count_mutex, NULL);
        pthread_cond_init (&count_condition, NULL);

        // for portability, explicity create threads in a joinable state
        pthread_t threads [NUM_THREADS];
        pthread_attr_t attr;
        pthread_attr_init (&attr);
        pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);

        printf("\nBeginning CPU simulation ...\n");
        double time1 = getTimeStamp();
        // creating the threads
        for (int i=0; i<NUM_THREADS; i++) {
                rc = pthread_create (&threads[i], &attr, computeHost_SMT, (void *) i);
                if (rc) {
                        printf("Error; return code from pthread_create() is %d.\n", rc);
                        exit(EXIT_FAILURE);
                }
        }

        // wait on the other threads
        for (int i=0; i<NUM_THREADS; i++) {
                rc = pthread_join (threads[i], &status);
                if (rc) {
                        printf("ERROR; return code from pthread_join() is %d.\n", rc);
                        exit(EXIT_FAILURE);
                }
                // printf("main(): completed join with thread #%ld having status of %ld\n",
                //  i, (long) status);
        }

        double time2 = getTimeStamp();
        double elapsedTime = time2-time1;

        pthread_attr_destroy (&attr);
        pthread_mutex_destroy (&count_mutex);
        pthread_cond_destroy (&count_condition);

        printf("\nElapsed Time: %lfs\n", elapsedTime);
        printf("Average timestep simulation duration: %lfs\n", elapsedTime/maxIteration);

        pthread_exit (NULL);

        // double time1 = getTimeStamp();
        // h_func(h_nbody);
        // double time2 = getTimeStamp();
        // double elapsedTime = time2-time1;
        // printf("Elapsed Time: %.4lfs\n", elapsedTime);
        // printf("Elapsed Time per Iteration: %.4lfs\n", elapsedTime/maxIteration);

        return 0;
}
