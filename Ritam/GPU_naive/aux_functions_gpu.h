#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

//Error handle
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort=true)
{
       if (code != cudaSuccess)
       {
         fprintf(stderr,"GPUassert: %s %s %d\n",
         cudaGetErrorString(code), file, line);
         if (abort) exit(code);
       }
}

// time stamp function in seconds 
double getTimeStamp() {     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL  ) ;     
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;  
} 