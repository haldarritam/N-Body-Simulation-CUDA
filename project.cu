#include <stdio.h>
#include<sys/time.h>

#define MAX_INITIAL_WEIGHT 1000
#define MAX_INITIAL_RANGE 10000
#define MAX_INITIAL_VELOCITY 100
#define BLOCK_DIM 32	

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
void h_acce(float4* A, float3* B, int noElems){
	float3 r;
	for (int i = 0; i < noElems; i++)
	    for (int j = 0; j < noElems; j++)
		    if (i != j){
        	    r.x = A[j].x - A[i].x;
                r.y = A[j].y - A[i].y;
                r.z = A[j].z - A[i].z;
				
				float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
                float distSixth = distSqr * distSqr * distSqr;
                float invDistCube = 1.0f/sqrtf(distSixth);
                float s = A[j].w * invDistCube;
				
                B[i].x += r.x * s;
                B[i].y += r.y * s;
                B[i].z += r.z * s;
	        }
}

// host-side preprocess the data
void h_preprocess(float4* A, float3 *V, float3* B, float dt, int noElems){
	h_acce(A, B, noElems);
	for (int i = 0; i < noElems; i++){		
		V[i].x += 0.5 * B[i].x * dt;
		V[i].y += 0.5 * B[i].y * dt;
		V[i].z += 0.5 * B[i].z * dt;
	}	
}

// host-side integration
void h_inte(float4* A, float3 *V, float3* B, float dt, int noElems){
	for (int i = 0; i < noElems; i++){
		A[i].x += V[i].x * dt;
		A[i].y += V[i].y * dt;
		A[i].z += V[i].z * dt;
	}
		
	h_acce(A, B, noElems);
		
	for (int i = 0; i < noElems; i++){
		
		V[i].x += B[i].x * dt;
		V[i].y += B[i].y * dt;
		V[i].z += B[i].z * dt;
	}	
}

// host-side function
void h_func(float4* A, float3 *V, float3* B, float dt, int noElems, int maxIteration){
	double timeStampA = getTimeStamp() ;
	
	h_preprocess(A, V, B, dt, noElems);
	int i = 1;
	
	while (i < maxIteration){
		i++;
		h_inte(A, V, B, dt, noElems);
		
		FILE *f = fopen("position.txt", "w");
        for (int j = 0; j < noElems; j++)
            fprintf(f, "%.6f %.6f %.6f\n", A[j].x, A[j].y, A[j].z);
        fclose(f);
	}
	
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

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
    float s = bj.w * invDistCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
    int i;
    extern __shared__ float4 sharedPosition[];
    for (i = 0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, sharedPosition[i], accel);
    }
    return accel;
}

// device-side acceleration compution
__device__ void d_acce(float4 *X, float3 *A, int noElems)
{
    extern __shared__ float4 sharedPosition[];
    float4 myPosition;
    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = X[gtid];
    for (i = 0, tile = 0; i < noElems; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
		if (idx < noElems) sharedPosition[threadIdx.x] = X[idx];
        __syncthreads();
        acc = tile_calculation(myPosition, acc);
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
	int i = 1;
	
	while (i < maxIteration){
		i++;
		
        d_inte<<<(noElems + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(d_X, d_V, d_A, dt, noElems);
        cudaDeviceSynchronize();
		
		cudaMemcpy( h_X, d_X, noElems * sizeof(float4), cudaMemcpyDeviceToHost );
		
		FILE *f = fopen("position.txt", "w");
        for (int j = 0; j < noElems; j++)
            fprintf(f, "%.6f %.6f %.6f\n", d_X[j].x, d_X[j].y, d_X[j].z);
        fclose(f);
	}
	
	double timeStampB = getTimeStamp() ;
	printf("%.6f\n", timeStampB - timeStampA);
}
int main( int argc, char *argv[] ) {
	
    // get program arguments
    if( argc != 3) {
        printf( "Error: wrong number of args\n" ) ;
        exit(0) ;
    }
    int nx = atoi( argv[1] ) ; // nx is the number of columns
    int ny = atoi( argv[2] ) ; // ny is the number of rows
    int noElems = 30000 ;
	float dt = 0.01;
	int maxIteration = 10;
    //int bytes = noElems * sizeof(int) ;
	
    // alloc memory host-side
    //int *h_A = (int *) malloc( bytes ) ;
	float4 *h_X;
	cudaError_t status = cudaMallocHost((void**)&h_X, noElems * sizeof(float4));
	if (status != cudaSuccess){
        printf("Error: allocating pinned host memory\n");
		exit(0);
	}
    float3 *h_A = (float3 *) malloc( noElems * sizeof(float3) ) ; 	
	float3 *h_V;
	status = cudaMallocHost((void**)&h_V, noElems * sizeof(float3));
	if (status != cudaSuccess){
        printf("Error: allocating pinned host memory\n");
		exit(0);
	}
	
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
	
    h_func(h_X, h_V, h_A, dt, noElems, maxIteration);
	d_func(d_X, d_V, d_A, h_X, dt, noElems, maxIteration);
	
	// verification
    
	// free GPU resources
    cudaFree( d_A ); 
	cudaFree( d_V );
    cudaDeviceReset() ;
}