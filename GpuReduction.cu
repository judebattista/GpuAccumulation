#include <iomanip>
#include <iostream>


//CUDA Kernel
//Maps one thread to each output space
//Reduces the array once by a factor of reductionFactor
//Assumption: we have enough threads to span the output array for our given reductionFactor
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
//Maps one thread to each output space
//Can reduce the array to a single value
//Note that this reductionFactor applies to the complete reduction, not a single step.
//Each thread works over a single section  of the array, so the memory access is poorly optimized
//Each thread reduces it's section of the array to a single value via multiple iterations, which is then stored in the output array/
//Much like the single step variant, but solves the problem of insufficient threads to span the array
__global__ void reduceArrayOutputMap(double* fullArray, double* reducedArray, int fullArraySize, int reductionFactori) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //if we are in our problem space
    if (tid < fullArraySize) {
        //Figure out how many threads are working on the array
        int numThreads = gridDim.x * blockDim.x;
        //If numThreads >= reductionFactor, this is just the single step version
        //Note that this check will be identical throughout a block, preventing divergence
        if (numThreads >= reductionFactor) {
            int startingIndex = reductionFactor * tid;
            double balsamic = 0;
            for (int ndx = 0; ndx < reductionFactor; ndx++) {
                balsamic += *(fullArray + ((startingIndex + ndx) % fullArraySize));
            }
            balsamic /= reductionFactor;
            *(reducedArray + tid) = balsamic;
        } else { //Otherwise each thread will need to perform multiple reductions
            //Based on the number of threads, figure out the size of the section each thread is responsible for
            int sectionLen = (fullArraySize + numThreads - 1) / numThreads;
            //Figure out where this thread should start its section
            int sectionStart = sectionLen * tid; 
            int reducedArraySize = (fullArraySize + reductionFactor - 1) / reductionFactor;
            double gravy = 0;
            for (int ndx = 0; ndx < reductionFactor; ndx++) {
                gravy += *(fullArray + ((startingIndex + ndx) % fullArraySize));
            }
            gravy /= reductionFactor;
            *(reducedArray + tid) = gravy;
        }
    }
}


//CUDA Kernel
//Can reduce the array to a single value
//Note that this reductionFactor applies to the complete reduction, not a single step.
//Each SET of threads spans a single segment, meaning that memory access for a given segment should be contiguous
//This method does not require having enough threads to span the entire array
__global__ void reduceArrayOutputMap(double* fullArray, double* reducedArray, int fullArraySize, int reductionFactori) {
    int numThreads = gridDim.x * blockDim.x;
    //We want each block of threads to address a specific section of the array of size N / numberOfBlocks
    //Within that section, we want the set of threads in the block to span a segment of size section / numberOfThreadsPerBlock
    //  and then move to another segment together until the section is complete
    //Once each thread as averaged its values, write those values to shared memory
    //Then thread 0 averages the values in shared memory
    //Possible optimization: have multiple threads average the values in shared memory
    //Need to __synchthreads() after the shared memory write
    int reducedArraySize = (fullArraySize + reductionFactor - 1) / reductionFactor;
    __shared__ double subReductions[reducedArraySize];
    
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
    reduceArraySingleStep<<<blocks, threadsPerBlock>>>(dev_fullArray, dev_reducedArray, fullArraySize, reductionFactor); 

    //Copy data from device to host
    cudaMemcpy(reducedArray, dev_reducedArray, reducedArraySize * sizeof(double), cudaMemcpyDeviceToHost);
   
    printArray(reducedArray, reducedArraySize);

    cudaFree(dev_fullArray);
    cudaFree(dev_reducedArray); 
}
