#ifndef BNB_CUDA_hpp
#define BNB_CUDA_hpp

#include <chrono>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <tsl/robin_map.h>
#include <igl/repmat.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <igl/repmat.h>
#include <igl/mat_min.h>
#include <cassert> 
#include <cuda_runtime.h>
#include "cost_time_ratio_solver.hpp"
#include <cmath>

#define CUSTOM_CUDA_INF 2147483647
#define custom_is_inf(x) x >= (CUSTOM_CUDA_INF-1)
#define DEBUG_CTR_CUDA true
#define THREAD_VERTEX_CYCLE_ENCOUNTER 1024

inline cudaDeviceProp initialize_gpu(const std::string prefix, const bool verbose) {
    int cuda_device = 0;
    //cudaGetDevice(&cuda_device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, cuda_device);
    cudaSetDevice(cuda_device);
    if (verbose) {
        std::cout << prefix<< "Going to use " << props.name << " " << props.major << "." << props.minor << ", device number " << cuda_device << "\n";
        std::cout << prefix << "  > maxThreadsPerBlock = " << props.maxThreadsPerBlock  << std::endl
                  << prefix << "  > multiProcessorCount = " << props.multiProcessorCount   << std::endl
                  << prefix << "  > maxThreadsPerMultiProcessor = "<< props.maxThreadsPerMultiProcessor << std::endl;
    }
    return props;
}

inline void checkCudaError(std::string place = "") {
    cudaDeviceSynchronize();
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cout << place <<" CUDA error: " <<cudaGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

inline void checkPointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes (&attributes, ptr);
    fprintf(stdout, "Memory type %d \n",(attributes).type);
    fprintf(stdout, "Device %d \n",(attributes).device);
}


namespace ctrsolver {
namespace cuda {
template<typename DTYPE>
class CudaMatrix {
    private:
    long nrows, ncols;
    bool rowMajor;
    DTYPE* data;
    public:
    CudaMatrix() {
        nrows = 0;
        ncols = 0;
        data = nullptr;
        rowMajor = true;
    }
    CudaMatrix(const Eigen::MatrixX<DTYPE>& X) {
        nrows = X.rows();
        ncols = X.cols();
        rowMajor = true;
        if (DEBUG_CTR_CUDA) {
            if (X.rows() != 1 && X.cols() != 1) {
                printf("Not handling conversion from row major to column major etc ");
                //  X.IsRowMajor;
            }
        }       
        cudaMalloc((void **)&data,  nrows * ncols * sizeof(DTYPE));
        if (DEBUG_CTR_CUDA) checkCudaError("malloc cudamatrix eigen constructur");
        cudaMemcpy(data, X.data(), nrows * ncols * sizeof(DTYPE), cudaMemcpyHostToDevice);
        if (DEBUG_CTR_CUDA) checkCudaError("memcpy cudamatrix eigen constructur");
    }
    CudaMatrix(const long rows, const long cols) {
        nrows = rows;
        ncols = cols;
        rowMajor = true;
        cudaMalloc((void **)&data,  nrows * ncols * sizeof(DTYPE));
        if (DEBUG_CTR_CUDA) checkCudaError("malloc cuda matrix rows cols constructur");
    }
    ~CudaMatrix() {
        if (DEBUG_CTR_CUDA) checkCudaError("cudafree cudamatrix destructor");
    }
    void destroy() {
        if (data != nullptr)
            cudaFree(data);
        if (DEBUG_CTR_CUDA) checkCudaError("destroy() cudamatrix destructor");
        ncols = 0;
        nrows = 0;
    }
    __device__ DTYPE& operator()(const int row = 0, const int col = 0) {
        assert(row >= 0 && row < nrows);
        assert(col >= 0 && col < ncols);
        if (DEBUG_CTR_CUDA) {
            if (row < 0) {printf("Rowidx negative\n"); return data[0];}
            if (col < 0) {printf("Colidx negative\n"); return data[0];}
            if (row >= nrows) {printf("Rowidx too large %d %d\n", row, nrows); return data[0];}
            if (col >= ncols) {printf("Colidx too large %d %d\n", col, ncols); return data[0];}
        }
        if (rowMajor) 
            return data[row * ncols + col];
        else 
            return data[row + col * nrows];
    }
    __device__ DTYPE operator()(const int row = 0, const int col = 0) const {
        assert(row >= 0 && row < nrows);
        assert(col >= 0 && col < ncols);
        if (DEBUG_CTR_CUDA) {
            if (row < 0) {printf("Rowidx negative\n"); return data[0];}
            if (col < 0) {printf("Colidx negative\n"); return data[0];}
            if (row >= nrows) {printf("Rowidx too large %d %d\n", row, nrows); return data[0];}
            if (col >= ncols) {printf("Colidx too large %d %d\n", col, ncols); return data[0];}
        }
        if (rowMajor) 
            return data[row * ncols + col];
        else 
            return data[row + col * nrows];
    }
    DTYPE getMatrixValueFromHostAt(const int row = 0, const int col = 0) {
        // this function is very costly => should not be called very often :)
        assert(row >= 0 && row < nrows);
        assert(col >= 0 && col < ncols);
        DTYPE hostData;
        if (rowMajor) 
            cudaMemcpy(&hostData, &data[row * ncols + col], sizeof(DTYPE), cudaMemcpyDeviceToHost);
        else 
            cudaMemcpy(&hostData, &data[row + col * nrows], sizeof(DTYPE), cudaMemcpyDeviceToHost);
        return hostData;
    }
    CudaMatrix& operator=(CudaMatrix other) {
        std::swap(this->nrows, other.nrows);
        std::swap(this->ncols, other.ncols);
        std::swap(this->data, other.data);
        std::swap(this->rowMajor, other.rowMajor);
        return *this;
    }
    void setConstant(const DTYPE constant) {
        const long sz = nrows * ncols * sizeof(DTYPE);
        thrust::device_ptr<DTYPE> dev_ptr(data);
        thrust::fill(dev_ptr, dev_ptr + (nrows * ncols), constant);
        if (DEBUG_CTR_CUDA) checkCudaError("memset setConstant");
    }
    __device__ __host__ void setRowConstant(const int row, const DTYPE constant) {
        if (DEBUG_CTR_CUDA && !rowMajor) printf("This function only works for row-major\n");
        thrust::device_ptr<DTYPE> dev_ptr(data);
        thrust::fill(dev_ptr + (row * ncols), dev_ptr + ((row + 1) * ncols), constant);
        if (DEBUG_CTR_CUDA) checkCudaError("memset setConstant");
    }
    void setBlockRowsConstant(const int startRow, const int numRows, const DTYPE constant) {
        if (DEBUG_CTR_CUDA && !rowMajor) printf("This function only works for row-major\n");
        thrust::device_ptr<DTYPE> dev_ptr(data);
        thrust::fill(dev_ptr + (startRow * ncols), dev_ptr + ((startRow + numRows) * ncols), constant);
        if (DEBUG_CTR_CUDA) checkCudaError("memset setConstant");
    }
    void copyRowToRow(const int srcRow, const int trgtRow) {
        if (DEBUG_CTR_CUDA && !rowMajor) printf("This function only works for row-major\n");
        const int size =  sizeof(DTYPE) * ncols;
        cudaMemcpy(&data[trgtRow * ncols], &data[srcRow * ncols], size, cudaMemcpyDeviceToDevice);
        if (DEBUG_CTR_CUDA) checkCudaError("copyrowtorow");
    }
    void setZero() {
        setConstant(0);
    }
    void setOnes() {
        setConstant(1);
    }
    // quite dangerous to change majorarity at runtime lol
    void setRowMajor() {
        rowMajor = true;
    }
    void setColMajor() {
        rowMajor = false;
    }
    void copyEigen(const Eigen::MatrixX<DTYPE>& X) {
        if (DEBUG_CTR_CUDA) {
            if (X.rows() != 1 && X.cols() != 1) {
                printf("Not handling conversion from row major to column major etc ");
                //  X.IsRowMajor;
            }
        }   
        assert(X.rows() == nrows);
        assert(X.cols() == ncols);
        cudaMemcpy(data, X.data(), nrows * ncols * sizeof(DTYPE), cudaMemcpyHostToDevice);
        if (DEBUG_CTR_CUDA) checkCudaError("cudamemcpy from eigen");
    }
    void copyToEigen(Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X) {
        if (DEBUG_CTR_CUDA) {
            if (!rowMajor) printf("This function is not designed for !rowMajor");
        }   
        X = Eigen::Matrix<DTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(nrows, ncols); 
        cudaMemcpy(X.data(), data, nrows * ncols * sizeof(DTYPE), cudaMemcpyDeviceToHost);
        if (DEBUG_CTR_CUDA) checkCudaError("cudamemcpy to eigen");
    }
    void check() {
        fprintf(stdout, "Data at %p with size %d %d\n",(void*) data, nrows, ncols);
        checkPointer((void*)data);
        checkCudaError();
    }
    __device__ long rows() const {
        return nrows;
    }
    __device__ long cols() const {
        return ncols;
    }
};

template<typename DTYPE>
__global__ void printCudaMatrix(const CudaMatrix<DTYPE> mat) {
    for (long i = 0; i < mat.rows(); i++) {
        for (long j = 0; j < mat.cols(); j++) {
            printf("%.2f ", (float) mat(i, j));
        }
        printf("\n");
    }
}

class VECSET {
private:
    bool isInited;
    long maxV;
    long size;
    long totalSize;
    CudaMatrix<int> dataarray;
    CudaMatrix<int> minIdxV;
    CudaMatrix<int> maxIdxV;
public:
    VECSET() {isInited = false;};
    VECSET(std::vector<tsl::robin_set<long>>& inp) {
        maxV = inp.size();
        totalSize = 0;
        printf("TODO: optimise vecset constructor\n");
        for (int i = 0; i < maxV; i++) {
            for (const auto& it : inp.at(i)) {
                totalSize++;
            }
        }
        Eigen::MatrixX<int> _data(totalSize, 1), _minIdxV(maxV, 1), _maxIdxV(maxV, 1);
        long idx = 0;
        for (int i = 0; i < maxV; i++) {
            _minIdxV(i) = idx;
            for (const auto& it : inp.at(i)) {
                const long e = it;
                _data(idx) = e;
                idx++;
            }
            _maxIdxV(i) = idx;
        }

        // copy data to device
        dataarray = CudaMatrix<int>(_data);
        minIdxV = CudaMatrix<int>(_minIdxV);
        maxIdxV = CudaMatrix<int>(_maxIdxV);
        isInited = true;
    }
    void destroy() {
        if (isInited) {
            dataarray.destroy();
            minIdxV.destroy();
            maxIdxV.destroy();
            isInited = false;
        }
    } 
    __device__ long atVminIdx(const long v) const {
        return minIdxV(v);
    }
    __device__ long atVmaxIdx(const long v) const {
        return maxIdxV(v);
    }
    __device__ long data(const long idx) const {
        return dataarray(idx);
    }
};

class CYCLE {
public:
    int* data;
    int* numElements;
    int maxSize;
    CYCLE () {maxSize = 0;}
    CYCLE(size_t size) {
        maxSize = size;
        int numElementsHost = 0;
        cudaMalloc((void **)&numElements,  sizeof(int));
        cudaMemcpy(numElements, &numElementsHost, sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&data,  size * sizeof(int));
        if (DEBUG_CTR_CUDA) checkCudaError("CYCLE constructor");
    }
    ~CYCLE() {
        //cudaFree(data);
    }
    __device__ int& operator()(int idx) {
        if (idx < *numElements && idx >= 0)
            return data[idx];
    }
    __device__ void push_back(int newval) {
        data[*numElements] = newval;
        (*numElements)++;
    }
    __device__ void clear() {
        (*numElements) = 0;
    }
    int size() {
        int numElementsHost;
        cudaMemcpy(&numElementsHost, numElements, sizeof(float), cudaMemcpyDeviceToHost);
        if (DEBUG_CTR_CUDA) checkCudaError("CYCLE size()");
        return numElementsHost;
    }
    void destroy() {
        cudaFree(data);
        cudaFree(numElements);
        if (DEBUG_CTR_CUDA) checkCudaError("CYCLE destroy()");
    }
};

typedef struct LBTUPLE {
    float minmeanlb;
    int k;
    int v;
    __device__ __host__ LBTUPLE() {}
    __device__ __host__ LBTUPLE(float _minmeanlb, int _k, int _v) {
        minmeanlb = _minmeanlb;
        k = _k;
        v = _v;
    }
    __device__ __host__ friend bool operator<(LBTUPLE const& a, LBTUPLE const& b){
        return a.minmeanlb < b.minmeanlb;
    }
} LBTUPLE;
//typedef std::tuple<float, int, int> LBTUPLE;
/*inline bool smaller(const LBTUPLE& a, LBTUPLE& b) {
    return a.minmeanlb < b.minmeanlb;
}
inline bool larger(const LBTUPLE& a, LBTUPLE& b) {
    return a.minmeanlb > b.minmeanlb;
}*/

class LBARRAY {
public:
    LBTUPLE* data;
    LBARRAY() {};
    LBARRAY(size_t size) {
        cudaMalloc((void **)&data,  size * sizeof(LBTUPLE));
        if (DEBUG_CTR_CUDA) checkCudaError("LBARRAY constructor");
    }
    ~LBARRAY() {
        //cudaFree(data);
    }
    __device__ __host__ LBTUPLE& operator()(int idx) {
        return data[idx];
    }
    void destroy() {
        cudaFree(data);
        if (DEBUG_CTR_CUDA) checkCudaError("LBARRAY destroy()");
    }
};

struct CTRGPUSTATE {
    CTRGPUSTATE(){};
    int branchId;
    int maxDepthPerLayer;
    Eigen::MatrixXi nodeBranchId;
    Eigen::MatrixX<PRECISION_CTR> cost;
    Eigen::MatrixX<PRECISION_CTR> time;
    CudaMatrix<int> SRCIds;
    CudaMatrix<PRECISION_CTR> costCuda;
    CudaMatrix<PRECISION_CTR> timeCuda;
    CudaMatrix<int> predecessor;
    CudaMatrix<PRECISION_CTR> DPtable;
    CudaMatrix<bool> vertexInCycleEncounter;
    CudaMatrix<int> cycleDetectionHelper;
    CudaMatrix<int> nodeBranchIdCuda;
    CudaMatrix<bool> anyPathImproves;
    std::vector<tsl::robin_set<long>> branchGraph;
    std::vector<tsl::robin_set<long>> vertex2preceedingEdgeMap;
    int numTotalNodes;
    std::string prefix;
    long numLayers;
    long numNodesPerLayer;
    long numNodes;
    PRECISION_CTR lowerBoundOfCurrentBranch;
    PRECISION_CTR lowerBound;
    PRECISION_CTR upperBound;
    PRECISION_CTR ratioLowerBoundOfCurrentBranch;
    PRECISION_CTR ratioLowerBound;
    PRECISION_CTR ratioUpperBound;
    std::vector<int> negativeCycle;
    std::vector<int> upperBoundCycle;
    std::vector<int> zeroLayerHits;
    int branchingSrcNode;
    int branchingTargetNode;
    int maxBranchId;
    bool upperboundFoundOnFirstTry;
    bool verbose;
    int numNodesInBranch;
    PRECISION_CTR currentRatio;
    PRECISION_CTR tolerance;
    Eigen::MatrixX<PRECISION_CTR> branchTable;
    CYCLE cycleCuda;
    VECSET vertex2preceedingEdgeMapCuda; // array of device vectors
    int numThreadBlocks;
    int threadsPerBlock;
};


/*


Function Headers


*/
BELLMANFORDSTATUS mooreBellmanFordSubroutineGPU(CTRGPUSTATE& ctrstate,
                                                const Eigen::MatrixXi& productspace,
                                                const Eigen::MatrixXi& SRCIds,
                                                const Eigen::MatrixXi& TRGTIds,
                                                const bool searchForZeroCycle=false,
                                                const bool searchForUpperBound=false);
bool lawlerSoubroutine(CTRGPUSTATE& ctrstate,
                       const Eigen::MatrixXi& productspace,
                       const Eigen::MatrixXi& SRCIds,
                       const Eigen::MatrixXi& TRGTIds);

std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> computeRatioForCurrentCycle(CTRGPUSTATE& ctrstate,
                                                                const Eigen::MatrixXi& productspace,
                                                                const Eigen::MatrixXi& SRCIds,
                                                                const Eigen::MatrixXi& TRGTIds,
                                                                const bool writeOutput=false);

std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> computeRatioForCycle(const CTRGPUSTATE& ctrstate,
                                                                               const std::vector<int> cycle,
                                                                               const Eigen::MatrixXi& productspace,
                                                                               const Eigen::MatrixXi& SRCIds,
                                                                               const Eigen::MatrixXi& TRGTIds,
                                                                               const bool writeOutput=false);
/*
template<typename MMC_COST_TYPE>
long setup(BNBSTATE<MMC_COST_TYPE>& bnbstate);

template<typename MMC_COST_TYPE>
float body(BNBSTATE<MMC_COST_TYPE>& bnbstate);

template<typename MMC_COST_TYPE>
bool tail(BNBSTATE<MMC_COST_TYPE>& bnbstate);*/


} // namespace cuda
} // namespace bnbcuda
#endif