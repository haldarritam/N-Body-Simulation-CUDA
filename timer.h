#include<stdio.h>
#include<sys/time.h>

// time stamp function in seconds 
double getTimeStamp() {     
    struct timeval  tv ;     
    gettimeofday( &tv, NULL  ) ;     
    return (double) tv.tv_usec/1000000 + tv.tv_sec ;  
} 