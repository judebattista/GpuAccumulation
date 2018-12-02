/*  Authors: Cameron Rutherford and Jude Battista
*   
*
*/

#include <iomanip>
#include <iostream>
#include <random>

//CUDA Kernel
//Maps one thread to each output space
//Reduces the array once by a factor of reductionFactor
//Assumption: we have enough threads to span the output array for our given reductionFactor
//This will be hideously inefficient as r approaches n
__global__ void reduceArraySingleStep(double* fullArray, double* reducedArray, int fullArraySize, int reductionFactor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //If we are in our problem space
    int reducedArraySize = (fullArraySize + reductionFactor - 1) / reductionFactor;
    if (tid < reducedArraySize) {
        //How far into the array should the thread start?
        int startingIndex = reductionFactor * tid;
        double balsamic = 0;
        for (int ndx = 0; ndx < reductionFactor; ndx++) {
            balsamic += *(fullArray + ((startingIndex + ndx) % fullArraySize));
        }
        balsamic /= reductionFactor;
        *(reducedArray + tid) = balsamic;
    }
}

//Goals: 
//No more than one shared memory load per item
//Optimize thread use
//Optimize memory coherence

//CUDA Kernel
//Single block reduction algorithm
//In order to use multiple blocks, the host can simply split the input array between blocks, call this kernel for each block, then recursively combine the results using this kernel 
__global__ void reduceBlock(double *idata, double *odata, unsigned int fullArraySize, int data_per_thread, int reducedArraySize) {

    //Shared data array. Holds intermediate reduction values
    extern __shared__ double sdata[];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int aggregate = 0;

    //Make a first pass, reducing the array to the number of threads
    for (int i = 0; i < data_per_thread; i++) {
        //Keep memory coalesced
        aggregate += idata[tid + (i * (blockDim.x)) % fullArraySize];
    }
    //store the results in shared memory, one output per thread
    sdata[tid] = double(aggregate) / data_per_thread;

    __syncthreads();

    //Now we need to reduce the values in shared memory
    //Start by reducing the values to the lowest power of two that is greater than our target size
    int i = 512; //length of shared data at the end of for loop
    for (; i > reducedArraySize - 1; i >>= 1) {
        if (tid < i) {
            sdata[tid] += sdata[tid + i];
            sdata[tid] /= 2;
        }
    }

    //Then make a single pass over the resulting array, wrapping around to pad the data as necessary
    if (tid < reducedArraySize) {
        odata[tid] = (sdata[tid] + sdata[(tid + reducedArraySize) % i]) / 2;
    }
}

//Fills an array with a repeating sequence of the integers from 1 to 100
//Primarily useful for testing. Does not generate randomly distributed data
void fillArray(double* target, int length) {
    for (int ndx = 0; ndx < length; ndx++) {
        *(target + ndx) = ndx % 100;
    }
}

void fillArrayRandom(double* target, int length) {
    std::mt19937 rng(time(0)); //Create a Mersenne Twister random number generator and seed it with the current time: https://www.guyrutenberg.com/2014/05/03/c-mt19937-example/
    std::uniform_int_distribution<std::mt19937::result_type>rand1to100(1, 100);
    for (int ndx = 0; ndx < length; ndx++) {
        *(target + ndx) = rand1to100(rng);
    }
}



//outputs an array in rows of 20 values
void printArray(double* target, int length) {
    for (int ndx = 0; ndx < length; ndx++) {
       std::cout << std::setw(7) << std::fixed << std::setprecision(2) << *(target + ndx);
       if (!((ndx+1) % 20)) {
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

int main() {
    double *fullArray, *reducedArray;
    double *dev_fullArray, *dev_reducedArray;
    int fullArraySize = 1<<25;
    int reductionFactor = 1<<24;
    //int reductionFactor = fullArraySize;
    int reducedArraySize = (fullArraySize + reductionFactor - 1) / reductionFactor;
    int threadsPerBlock = 1024;
    int blocks = 1;

    //unsigned int n = 100;//# of integers
    //unsigned int r = 100; //reduction factor
    //int q = (n + (r - 1)) / r; //number of elements in the final array
    int data_per_thread = (fullArraySize + threadsPerBlock - 1) / threadsPerBlock;
    int smSize = threadsPerBlock*sizeof(double);//shared mem

    //Timing variables
    cudaEvent_t start, stop; 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedMs;

    //Allocate host memory
    fullArray = (double *)malloc(sizeof(double) * fullArraySize);
    reducedArray = (double *)malloc(sizeof(double) * reducedArraySize);
    
    //Allocate device memory
    cudaMalloc((void**)&dev_fullArray, fullArraySize * sizeof(double));
    cudaMalloc((void**)&dev_reducedArray, reducedArraySize * sizeof(double));

    fillArrayRandom(fullArray, fullArraySize);
    //printArray(fullArray, fullArraySize); 

    //Test the naive kernel
    //Copy data from host to device
    cudaMemcpy(dev_fullArray, fullArray, fullArraySize * sizeof(double), cudaMemcpyHostToDevice);
    //start timer
    cudaEventRecord(start);
    //Run naive kernel
    reduceArraySingleStep<<<blocks, threadsPerBlock>>>(dev_fullArray, dev_reducedArray, fullArraySize, reductionFactor); 
    //stop timer 
    cudaEventRecord(stop);
    //Copy data from device to host
    cudaMemcpy(reducedArray, dev_reducedArray, reducedArraySize * sizeof(double), cudaMemcpyDeviceToHost);
    //Calculate and display elapsed time
    cudaEventElapsedTime(&elapsedMs, start, stop);
    std::cout << "The naive kernel took " << elapsedMs << "ms to reduce " << fullArraySize << " doubles down to " << reducedArraySize << " doubles.\n";
    printArray(reducedArray, reducedArraySize);

    //Test the single block kernel 
    //Copy data from host to device
    cudaMemcpy(dev_fullArray, fullArray, fullArraySize * sizeof(double), cudaMemcpyHostToDevice);
    //start timer
    cudaEventRecord(start);
    //Run single block kernel
    reduceBlock<<<blocks, threadsPerBlock, smSize>>>(dev_fullArray, dev_reducedArray, fullArraySize, data_per_thread, reducedArraySize); 
    //stop timer 
    cudaEventRecord(stop);
    //Copy data from device to host
    cudaMemcpy(reducedArray, dev_reducedArray, reducedArraySize * sizeof(double), cudaMemcpyDeviceToHost);
    //Calculate and display elapsed time
    cudaEventElapsedTime(&elapsedMs, start, stop);
    std::cout << "The single block kernel took " << elapsedMs << "ms to reduce " << fullArraySize << " doubles down to " << reducedArraySize << " doubles.\n";
 
    printArray(reducedArray, reducedArraySize);

    cudaFree(dev_fullArray);
    cudaFree(dev_reducedArray); 
}
