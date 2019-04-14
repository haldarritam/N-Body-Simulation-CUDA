#include <stdio.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX_INITIAL_WEIGHT 1000
#define MAX_INITIAL_RANGE 10000
#define MAX_INITIAL_VELOCITY 100
#define EPS 1e-9f
#define BLOCK_DIM 1024
#define G 8
#define DT 0.001
#define _5DT 0.0005

// time stamp function in seconds
double getTimeStamp() {
        struct timeval tv;
        gettimeofday( &tv, NULL );
        return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

//initialize velocity and position data
void initData(float4 *A, float3 *V, int noElems){
        for (int i = 0; i < noElems; i++) {
                A[i].w = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_WEIGHT);
                A[i].x = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);
                A[i].y = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_RANGE);

                V[i].x = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
                V[i].y = (float) rand() / (float) (RAND_MAX / MAX_INITIAL_VELOCITY);
        }
}

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
        float3 r;

        r.x = bj.x - bi.x;
        r.y = bj.y - bi.y;

        float distSqr = r.x * r.x + r.y * r.y + EPS;
        float distSixth = distSqr * distSqr * distSqr;
        float invDistCube = rsqrtf(distSixth);
        float s = bj.w * invDistCube;

        ai.x += r.x * s;
        ai.y += r.y * s;
        return ai;
}

// device-side acceleration compution
__device__ void d_acce(float4 *X, float3 *A, int noElems)
{
        int gtid = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ float4 sharedPosition[BLOCK_DIM];
        float4 myPosition;
        int i, j, tile;
        float3 acc = {0.0f, 0.0f, 0.0f};
        myPosition = X[gtid];
        for (i = 0, tile = 0; i < noElems; i += blockDim.x, tile++) {
                int idx = tile * blockDim.x + threadIdx.x;
                sharedPosition[threadIdx.x] = X[idx];
                __syncthreads();
                #pragma unroll (16)
                for (j = 0; j < blockDim.x; j++) {
                        acc = bodyBodyInteraction(myPosition, sharedPosition[j], acc);
                }
                __syncthreads();
        }
        // Save the result in global memory for the integration step.
        A[gtid] = acc;
}

// device-side preprocess the data
__global__ void d_preprocess(float4* X, float3 *V, float3* A, float dt, int noElems){
        d_acce(X, A, noElems);

        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < noElems) {
                V[i].x += 0.5 * A[i].x * dt;
                V[i].y += 0.5 * A[i].y * dt;
        }
}

// device-side integration
__global__ void d_inte(float4* X, float3 *V, float3* A, float dt, int noElems){
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < noElems) {
                X[i].x += V[i].x * dt;
                X[i].y += V[i].y * dt;

                d_acce(X, A, noElems);

                V[i].x += A[i].x * dt;
                V[i].y += A[i].y * dt;
        }
}

// device-side function
void d_func(float4* d_X, float3 *d_V, float3* d_A, float4 *h_X, float dt, int noElems, int maxIteration){
        double timeStampA = getTimeStamp();

        d_preprocess<<<(noElems + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(d_X, d_V, d_A, dt, noElems);
        int i = 0;

        //FILE *f = fopen("position2.txt", "w");
        while (i < maxIteration) {
                i++;

                d_inte<<<(noElems + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(d_X, d_V, d_A, dt, noElems);
                cudaDeviceSynchronize();

                //cudaMemcpy( h_X, d_X, noElems * sizeof(float4), cudaMemcpyDeviceToHost );

                //for (int j = 0; j < noElems; j++)
                        //fprintf(f, "%.6f %.6f %.6f\n", h_X[j].x, h_X[j].y, h_X[j].z);
        }
        //fclose(f);

        double timeStampB = getTimeStamp();
        printf("%.6f\n", timeStampB - timeStampA);
}

// !!!!!!!!!! Deal with G with weight
// !!!!!!!!!! fill the block with 0 if noElems cannot be divided by blocksize


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
        float4 *h_X;
        cudaError_t status = cudaMallocHost((void**)&h_X, noElems * sizeof(float4));
        if (status != cudaSuccess) {
                printf("Error: allocating pinned host memory\n");
                exit(0);
        }
        float3 *h_V;
        status = cudaMallocHost((void**)&h_V, noElems * sizeof(float3));
        if (status != cudaSuccess) {
                printf("Error: allocating pinned host memory\n");
                exit(0);
        }


        // init matrices with random data
        initData(h_X, h_V, noElems);

        // alloc memory dev-side
        float4 *d_X;
        float3 *d_A, *d_V;
        cudaMalloc( (void **) &d_X, noElems * sizeof(float4) );
        cudaMalloc( (void **) &d_A, noElems * sizeof(float3) );
        cudaMalloc( (void **) &d_V, noElems * sizeof(float3) );

        //transfer data to dev
        cudaMemcpy( d_X, h_X, noElems * sizeof(float4), cudaMemcpyHostToDevice );
        cudaMemcpy( d_V, h_V, noElems * sizeof(float3), cudaMemcpyHostToDevice );

        d_func(d_X, d_V, d_A, h_X, dt, noElems, maxIteration);

        // free GPU resources
        cudaFree( d_X );
        cudaFree( d_A );
        cudaFree( d_V );
        cudaDeviceReset();
}
