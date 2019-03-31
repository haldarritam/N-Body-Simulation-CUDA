#include <stdio.h>
#include<sys/time.h>
#include <pthread.h>

#define MAX_INITIAL_WEIGHT 1000
#define MAX_INITIAL_RANGE 10000
#define MAX_INITIAL_VELOCITY 100
#define EPS 1e-9f
#define NUM_THREADS 32
#define BLOCK_DIM 32
#define G 100

pthread_mutex_t mutex_tid;

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
	float4 *X = nbody -> X;
	int noElems = nbody -> noElems;
	int maxIteration = nbody -> maxIteration;
	
	double timeStampA = getTimeStamp();
	
	pthread_t *threads = (pthread_t*) malloc(NUM_THREADS * sizeof(pthread_t));
	
	h_preprocess(nbody, threads);
	int i = 0;
	
	FILE *f = fopen("position.txt", "w");
	while (i < maxIteration){
		i++;
		h_inte(nbody, threads);
		
        for (int j = 0; j < noElems; j++)
            fprintf(f, "%.6f %.6f %.6f\n", X[j].x, X[j].y, X[j].z);
	}
	fclose(f);
	
	double timeStampB = getTimeStamp() ;
	printf("%.6f\n", timeStampB - timeStampA);
}

// verification
bool verification(int* A, int* B, int noElems){
	for (int i = 0; i < noElems; i++)
        if (B[i] != A[i]) return false;
	return true;
}

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
    float3 r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS;
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    float s = bj.w * invDistCube * G;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ float3 tile_calculation(float4 myPosition, float3 accel, int sharedPositionLength)
{
    int i;
    __shared__ float4 sharedPosition[BLOCK_DIM];
    for (i = 0; i < sharedPositionLength; i++) {
        accel = bodyBodyInteraction(myPosition, sharedPosition[i], accel);
    }
    return accel;
}

// device-side acceleration compution
__device__ void d_acce(float4 *X, float3 *A, int noElems)
{
    __shared__ float4 sharedPosition[BLOCK_DIM];
    float4 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gtid >= noElems) return;
    myPosition = X[gtid];
    for (i = 0, tile = 0; i < noElems; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
		if (idx < noElems) sharedPosition[threadIdx.x] = X[idx];
		int sharedPositionLength = noElems - tile * blockDim.x;
		if (sharedPositionLength > blockDim.x) sharedPositionLength = blockDim.x;
        __syncthreads();
        acc = tile_calculation(myPosition, acc, sharedPositionLength);
        __syncthreads();
    }
    // Save the result in global memory for the integration step.
    A[gtid] = acc;
}

// device-side preprocess the data
__global__ void d_preprocess(float4* X, float3 *V, float3* A, float dt, int noElems){
	d_acce(X, A, noElems);
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < noElems){		
		V[i].x += 0.5 * A[i].x * dt;
		V[i].y += 0.5 * A[i].y * dt;
		V[i].z += 0.5 * A[i].z * dt;
	}	
}

// device-side integration
__global__ void d_inte(float4* X, float3 *V, float3* A, float dt, int noElems){	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < noElems){
	    X[i].x += V[i].x * dt;
	    X[i].y += V[i].y * dt;
	    X[i].z += V[i].z * dt;
		
	    d_acce(X, A, noElems);
		
	    V[i].x += A[i].x * dt;
	    V[i].y += A[i].y * dt;
	    V[i].z += A[i].z * dt;
	}
}

// device-side function
void d_func(float4* d_X, float3 *d_V, float3* d_A, float4 *h_X, float dt, int noElems, int maxIteration){
	double timeStampA = getTimeStamp() ;
	
	d_preprocess<<<(noElems + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(d_X, d_V, d_A, dt, noElems);
	int i = 0;
	
	FILE *f = fopen("position2.txt", "w");
	while (i < maxIteration){
		i++;
		
        d_inte<<<(noElems + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(d_X, d_V, d_A, dt, noElems);
        cudaDeviceSynchronize();
		
		cudaMemcpy( h_X, d_X, noElems * sizeof(float4), cudaMemcpyDeviceToHost );
		
		//FILE *f = fopen("position2.txt", "w");
        for (int j = 0; j < noElems; j++)
            fprintf(f, "%.6f %.6f %.6f\n", h_X[j].x, h_X[j].y, h_X[j].z);
        //fclose(f);
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
	float4 *h_X;
	cudaError_t status = cudaMallocHost((void**)&h_X, noElems * sizeof(float4));
	if (status != cudaSuccess){
        printf("Error: allocating pinned host memory\n");
		exit(0);
	}
    float3 *h_A = (float3 *) malloc( noElems * sizeof(float3) ); 	
	float3 *h_V;
	status = cudaMallocHost((void**)&h_V, noElems * sizeof(float3));
	if (status != cudaSuccess){
        printf("Error: allocating pinned host memory\n");
		exit(0);
	}
	nbodyStruct *h_nbody = (nbodyStruct*) malloc(sizeof(nbodyStruct));
	h_nbody->noElems = noElems;
	h_nbody->maxIteration = maxIteration;
	h_nbody->dt = dt;
	h_nbody->X = h_X;
	h_nbody->A = h_A;
	h_nbody->V = h_V;
	
	
    // init matrices with random data
    initData(h_X, h_V, noElems) ; 

    // alloc memory dev-side
	float4 *d_X;
    float3 *d_A, *d_V;
	cudaMalloc( (void **) &d_X, noElems * sizeof(float4) ) ;
    cudaMalloc( (void **) &d_A, noElems * sizeof(float3) ) ;
    cudaMalloc( (void **) &d_V, noElems * sizeof(float3) ) ;
	
    //transfer data to dev
	cudaMemcpy( d_X, h_X, noElems * sizeof(float4), cudaMemcpyHostToDevice );
    cudaMemcpy( d_V, h_V, noElems * sizeof(float3), cudaMemcpyHostToDevice );
	
    h_func(h_nbody);
	d_func(d_X, d_V, d_A, h_X, dt, noElems, maxIteration);
	
	// verification
    
	// free GPU resources
    cudaFree( d_A ); 
	cudaFree( d_V );
    cudaDeviceReset() ;
}
