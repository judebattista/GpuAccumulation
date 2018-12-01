#include <iomanip>
#include <iostream>


//CUDA Kernel
//Maps one thread to each output space
//Should not be optimal for large reductionFactors
__global__ void reduceArrayOutputMap(double* fullArray, double* reducedArray, int fullArraySize, int reductionFactor) {
    //int reducedArraySize = (fullArraySize + reductionFactor - 1) / reductionFactor;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //How far into the array should the thread start?
    int startingIndex = reductionFactor * tid;
    double balsamic = 0;
    for (int ndx = 0; ndx < reductionFactor; ndx++) {
        balsamic += *(fullArray + ((startingIndex + ndx) % fullArraySize));
    }
    balsamic /= reductionFactor;
    *(reducedArray + tid) = balsamic;
}

void fillArray(double* target, int length) {
    for (int ndx = 0; ndx < length; ndx++) {
        *(target + ndx) = ndx;
    }
}

void printArray(double* target, int length) {
    for (int ndx = 0; ndx < length; ndx++) {
       std::cout << std::setw(7) << std::fixed << std::setprecision(2) << *(target + ndx);
    }
    std::cout << "\n";
}

int main() {
    double *fullArray, *reducedArray;
    double *dev_fullArray, *dev_reducedArray;
    int reductionFactor = 2;
    int fullArraySize = 6;
    int reducedArraySize = (fullArraySize + reductionFactor - 1) / reductionFactor;
    int threadsPerBlock = reducedArraySize;
    int blocks = 1;

    //Allocate host memory
    fullArray = (double *)malloc(sizeof(double) * fullArraySize);
    reducedArray = (double *)malloc(sizeof(double) * reducedArraySize);
    
    //Allocate device memory
    cudaMalloc((void**)&dev_fullArray, fullArraySize * sizeof(double));
    cudaMalloc((void**)&dev_reducedArray, reducedArraySize * sizeof(double));

    fillArray(fullArray, fullArraySize);
    printArray(fullArray, fullArraySize); 

    //Copy data from host to device
    cudaMemcpy(dev_fullArray, fullArray, fullArraySize * sizeof(double), cudaMemcpyHostToDevice);

    //Run kernel
    reduceArrayOutputMap<<<blocks, threadsPerBlock>>>(dev_fullArray, dev_reducedArray, fullArraySize, reductionFactor); 

    //Copy data from device to host
    cudaMemcpy(reducedArray, dev_reducedArray, reducedArraySize * sizeof(double), cudaMemcpyDeviceToHost);
   
    printArray(reducedArray, reducedArraySize);

    cudaFree(dev_fullArray);
    cudaFree(dev_reducedArray); 
}
