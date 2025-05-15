#include "dijkstra_bnb_subroutine.cuh"
#include <algorithm>

namespace dijkstra {
namespace cuda {

// see https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893
__device__ void acquire_semaphore(volatile int *lock) {
    while (atomicCAS((int *)lock, 0, 1) != 0);
}
__device__ void release_semaphore(volatile int *lock) {
    *lock = 0;
    __threadfence_block();
}
int getOriginalGraphIndex(const DKSTRAGPUSTATE& dijkstrastate, const int pred) {
    const int numNodesPerLayerWithHelperLayers = dijkstrastate.numNodesPerLayer * dijkstrastate.maxDepthPerLayer;
    const int cycleLayer = pred / numNodesPerLayerWithHelperLayers;
    const int predInHelperLayer = (pred - cycleLayer * numNodesPerLayerWithHelperLayers);
    const int cycleHelperLayer = predInHelperLayer / dijkstrastate.numNodesPerLayer ;
    const int predInNormalLayer = predInHelperLayer - cycleHelperLayer * dijkstrastate.numNodesPerLayer;
    const int indexInOriginalGraph = predInNormalLayer + cycleLayer * dijkstrastate.numNodesPerLayer;
    if (DEBUG_DIJKSTRA_SOLVER && indexInOriginalGraph < 0 )
        std::cout << "indexInOriginalGraph < 0" << std::endl;
    return indexInOriginalGraph;
}
__device__ int getOriginalGraphIndexCuda(   const int numNodesPerLayer,
                                            const int maxDepthPerLayer,
                                            const int pred) {
    const int numNodesPerLayerWithHelperLayers = numNodesPerLayer * maxDepthPerLayer;
    const int cycleLayer = pred / numNodesPerLayerWithHelperLayers;
    const int predInHelperLayer = (pred - cycleLayer * numNodesPerLayerWithHelperLayers);
    const int cycleHelperLayer = predInHelperLayer / numNodesPerLayer ;
    const int predInNormalLayer = predInHelperLayer - cycleHelperLayer * numNodesPerLayer;
    const int indexInOriginalGraph = predInNormalLayer + cycleLayer * numNodesPerLayer;
    if (DEBUG_DIJKSTRA_SOLVER && indexInOriginalGraph < 0 )
        printf("indexInOriginalGraph < 0");
    return indexInOriginalGraph;
}


__global__ void setup_dp_table_cuda(CudaMatrix<PRECISION_DIJK> DPTABLE,
                                    CudaMatrix<int> PREDECESSOR,
                                    const CudaMatrix<int> NODEBRANCHID,
                                    int* numNodesInBranch,
                                    int* indexNodeInLayer,
                                    const int externalIter,
                                    const int branchId,
                                    const long numNodes,
                                    const long numNodesPerLayer) {
    const long i = threadIdx.x + blockIdx.x * blockDim.x + externalIter * (blockDim.x * gridDim.x);;
    if (i >= numNodesPerLayer) return;
    const int startVertexId = i;
    if (NODEBRANCHID(i) == branchId) {
        DPTABLE(0, startVertexId) = 0;
        PREDECESSOR(startVertexId) = -1;
        *indexNodeInLayer = i; // no threadfencing or shit like this necessariy as we only want to know ANY node index in layer
        atomicAdd(numNodesInBranch, 1);
    }
}


__global__ void body_cuda(CudaMatrix<PRECISION_DIJK> DPTABLE, 
                          CudaMatrix<int> PREDECESSOR, 
                          CudaMatrix<bool> anyPathImproves,
                          const CudaMatrix<int> SRCIds,
                          const CudaMatrix<PRECISION_DIJK> cost,
                          const PRECISION_DIJK currentRatio,
                          const PRECISION_DIJK upperBound,
                          const VECSET vertex2preceedingEdgeMap,
                          const long layer,
                          const long maxDepthPerLayer,
                          const long numNodesPerLayer,
                          const long numLayers,
                          const int externalIter,
                          const int kOffsetPreviousLayer,
                          const int kOffsetCurrentLayer,
                          const int k) {
    const int blockIndex = blockIdx.x;
    const int threadIndex = threadIdx.x + blockIndex * blockDim.x + externalIter * (blockDim.x * gridDim.x);
    if (threadIndex >= numNodesPerLayer) return;
    const int vi = threadIndex;
    const int threadId = 0;
    const int v = vi + layer * numNodesPerLayer * maxDepthPerLayer + k * numNodesPerLayer;
    const int actualV = vi + layer * numNodesPerLayer;
    for (long idx = vertex2preceedingEdgeMap.atVminIdx(actualV); idx < vertex2preceedingEdgeMap.atVmaxIdx(actualV); idx++) {
        const long e = vertex2preceedingEdgeMap.data(idx);
        if (e == -1) {
            continue;
        }
        PRECISION_DIJK newCost = cost(e);
        long srcIdx = SRCIds(e);
        const bool isIntraLayerEdge = (srcIdx / numNodesPerLayer) == layer;

        if (isIntraLayerEdge && k != 0) {
            if (maxDepthPerLayer <= 1) continue;
            const int srcIdxInLayer = srcIdx - layer * numNodesPerLayer;
            if (custom_is_inf(DPTABLE(kOffsetCurrentLayer + k - 1, srcIdxInLayer))) continue;

            const PRECISION_DIJK newVal = DPTABLE(kOffsetCurrentLayer + k - 1, srcIdxInLayer) + newCost;
            if ( newVal < DPTABLE(kOffsetCurrentLayer + k, vi) && newVal < upperBound) {
                anyPathImproves(0) = true; // this is safe https://stackoverflow.com/questions/5953955/concurrent-writes-in-the-same-global-memory-location
                const int predecessorIndex = srcIdxInLayer + layer * numNodesPerLayer * maxDepthPerLayer + (k-1) * numNodesPerLayer;
                PREDECESSOR(v) = predecessorIndex;
                DPTABLE(kOffsetCurrentLayer + k, vi) = newVal;
            }
        }
        if (!isIntraLayerEdge && k == 0) { // we need to connect to all layer depth extensions
            for (int kk = 0; kk < maxDepthPerLayer; kk++) {
                const int srcIdxInLayer = srcIdx - (layer-1) * numNodesPerLayer;
                if (custom_is_inf(DPTABLE(kOffsetPreviousLayer + kk, srcIdxInLayer))) continue;

                const PRECISION_DIJK newVal = DPTABLE(kOffsetPreviousLayer + kk, srcIdxInLayer) + newCost;
                if ( newVal < DPTABLE(kOffsetCurrentLayer, vi) && newVal < upperBound) {
                    anyPathImproves(0) = true; // this is safe, see comment above
                    const int predecessorIndex = srcIdxInLayer + (layer-1) * numNodesPerLayer * maxDepthPerLayer + kk * numNodesPerLayer;
                    PREDECESSOR(v) = predecessorIndex;
                    DPTABLE(kOffsetCurrentLayer, vi) = newVal;
                }
            }
        }
    }
}

__global__ void extract_cycle(const CudaMatrix<int> PREDECESSOR,
                              CYCLE cycle,
                              int* pred,
                              const int numNodesPerLayer,
                              const int maxDepthPerLayer,
                              const int maxLength,
                              const int vOnLastLayer) {
    cycle.clear();
    int predBackup = *pred;
    *pred = PREDECESSOR(vOnLastLayer);
    cycle.push_back(getOriginalGraphIndexCuda(numNodesPerLayer, maxDepthPerLayer, *pred));
    for (int kk = maxLength; kk > 0; kk--) {
        if (*pred == -1) break;
        predBackup = *pred;
        *pred = PREDECESSOR(*pred);
        if (*pred == -1) break;
        cycle.push_back(getOriginalGraphIndexCuda(numNodesPerLayer, maxDepthPerLayer, *pred));
    }
    *pred = predBackup;
}

__global__ void copyZeroLayer(  CudaMatrix<PRECISION_DIJK> DPTABLE, 
                                CudaMatrix<int> PREDECESSOR, 
                                CudaMatrix<bool> anyPathImproves,
                                const CudaMatrix<int> nodeBranchId, 
                                const int branchId,
                                const long maxDepthPerLayer,
                                const long numNodesPerLayer,
                                const long numLayers,
                                const int externalIter,
                                const int kIndexLastLayer, 
                                const int kIndexZeroLayerBackup) {
    const int blockIndex = blockIdx.x;
    const int threadIndex = threadIdx.x + blockIndex * blockDim.x + externalIter * (blockDim.x * gridDim.x);
    if (threadIndex >= numNodesPerLayer) return;
    const int vi = threadIndex;

    if (nodeBranchId(vi) != branchId) return;
    const int vOnLastLayer = numNodesPerLayer * maxDepthPerLayer * numLayers + vi;
    if (DPTABLE(kIndexLastLayer, vi) < DPTABLE(kIndexZeroLayerBackup, vi)) {
        anyPathImproves(0) = true; // this is save, see above
        DPTABLE(0, vi) = DPTABLE(kIndexLastLayer, vi);
        PREDECESSOR(vi) = PREDECESSOR(vOnLastLayer); // it is not "vOnLastLayer" as we would have zero layer index twice then
    }
    else {
        DPTABLE(0, vi) = DPTABLE(kIndexZeroLayerBackup, vi);
    }
}

__global__ void extract_lower_bounds(const CudaMatrix<PRECISION_DIJK> DPTABLE,
                                        const CudaMatrix<int> nodeBranchId,
                                        LBARRAY lbarray,
                                        const int branchId,
                                        const int maxDepthPerLayer,
                                        const long numNodesPerLayer,
                                        const long numLayers,
                                        const int externalIter,
                                        const int kIndexLastLayer,
                                        bool* anyPotentialNewLowerBound) {
    const int blockIndex = blockIdx.x;
    const int threadIndex = threadIdx.x + blockIndex * blockDim.x + externalIter * (blockDim.x * gridDim.x);
    if (threadIndex >= numNodesPerLayer) return;
    const long i = threadIndex;
    lbarray(i) = LBTUPLE(CUSTOM_CUDA_INF, i);
    if (nodeBranchId(i) != branchId) return;

    const PRECISION_DIJK lbcandidate = DPTABLE(kIndexLastLayer, i);
    if (!custom_is_inf(lbcandidate)) {
        *anyPotentialNewLowerBound = true;
    }
    lbarray(i) = LBTUPLE(lbcandidate, i);
}


__global__ void check_for_cycles(   const CudaMatrix<PRECISION_DIJK> DPTABLE, 
                                    const CudaMatrix<int> PREDECESSOR, 
                                    const CudaMatrix<int> nodeBranchId,
                                    CudaMatrix<bool> vertexInCycleEncounter,
                                    CudaMatrix<int> cycleDetectionHelper,
                                    const int branchId,
                                    const long numNodesPerLayer,
                                    const int externalIter,
                                    const PRECISION_DIJK tolerance,
                                    bool* existsNegativeCycle,
                                    const bool searchForZeroCycle,
                                    const int maxCycleLength) {
    const int blockIndex = blockIdx.x;
    const int threadIndex = threadIdx.x + blockIndex * blockDim.x + externalIter * (blockDim.x * gridDim.x);
    if (threadIndex >= numNodesPerLayer) return;
    const int vi = threadIndex;
    const PRECISION_DIJK lowerBound = DPTABLE(0, vi);
    if (nodeBranchId(vi) != branchId) return;
    if (!searchForZeroCycle && lowerBound >= 0) {
        return;
    }
    if (searchForZeroCycle && (lowerBound <= -tolerance || lowerBound >= tolerance) ) {
        return;
    }
    const int threadId = threadIdx.x;
    int pred = PREDECESSOR(vi);
    vertexInCycleEncounter(threadId, vi) = true;
    int cLength = 1;
    for (int kk = maxCycleLength - 1; kk > 0; kk--) {
        if (pred < numNodesPerLayer && pred >= 0) {
            if (vertexInCycleEncounter(threadId, pred)) { // we already have seen this vertex
                {
                    *existsNegativeCycle = true; // this is safe
                    cycleDetectionHelper(threadId, 0) = pred;
                    cycleDetectionHelper(threadId, 1) = cLength;
                    cycleDetectionHelper(threadId, 2) = (int)lowerBound;
                }
                break;
            }
            vertexInCycleEncounter(threadId, pred) = true;
        }
        pred = PREDECESSOR(pred);
        if (pred == -1) break;
        cLength++;
    }
}

/*


BODY


*/
bool mooreBellmanFordSubroutineGPU(DKSTRAGPUSTATE& dijkstrastate,
                                    const Eigen::MatrixXi& productspace,
                                    const Eigen::MatrixXi& SRCIds,
                                    const Eigen::MatrixXi& TRGTIds,
                                    const bool searchForUpperBound) {
    const int externalIterOffset = dijkstrastate.numThreadBlocks * dijkstrastate.threadsPerBlock;
    const int nBlocks = dijkstrastate.numThreadBlocks;
    const int nThreads = dijkstrastate.threadsPerBlock;
    const int numThreads = 1;
    // setup
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    Eigen::MatrixXf perVertexLowerBounds(dijkstrastate.numNodesPerLayer, numThreads);
    perVertexLowerBounds.setConstant(CUSTOM_CUDA_INF);
    Eigen::MatrixXi perVertexLbK(dijkstrastate.numNodesPerLayer, numThreads);
    dijkstrastate.DPtable.setConstant(CUSTOM_CUDA_INF);

    int indexNodeInLayer = -1;
    int numNodesInBranch = 0;
    int* num_nodes_in_branch_device;
    int* index_node_in_layer_device;
    cudaMalloc((void **)&num_nodes_in_branch_device,  1 * sizeof(int));
    cudaMalloc((void **)&index_node_in_layer_device,  1 * sizeof(int));
    cudaMemcpy(num_nodes_in_branch_device, &numNodesInBranch, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(index_node_in_layer_device, &indexNodeInLayer, sizeof(int), cudaMemcpyHostToDevice);
    int externalIter = 0;
    for (int i = 0; i < dijkstrastate.nodeBranchId.rows(); i += externalIterOffset) {
        setup_dp_table_cuda<<<nBlocks, nThreads>>>( dijkstrastate.DPtable, 
                                                    dijkstrastate.predecessor, 
                                                    dijkstrastate.nodeBranchIdCuda, 
                                                    num_nodes_in_branch_device,
                                                    index_node_in_layer_device,
                                                    externalIter,
                                                    dijkstrastate.branchId,
                                                    dijkstrastate.numNodes,
                                                    dijkstrastate.numNodesPerLayer);
        externalIter++;
    }
    if (DEBUG_CTR_CUDA) checkCudaError("setup end: ");
    cudaMemcpy(&numNodesInBranch, num_nodes_in_branch_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&indexNodeInLayer, index_node_in_layer_device, sizeof(int), cudaMemcpyDeviceToHost);
    if (dijkstrastate.verbose) std::cout << dijkstrastate.prefix << "      > Num nodes in branch " << numNodesInBranch << std::endl;
    dijkstrastate.numNodesInBranch = numNodesInBranch;
    if (numNodesInBranch == 0) {
        std::cout << "NO NODES IN BRANCH, this should not happen" << std::endl; exit(-1);
    }


    // dynamic programming loop: sweep through all layers
    bool negativeCycleFound = false;
    int iter = 0;
    const int kIndexZeroLayerBackup  = 2 * dijkstrastate.maxDepthPerLayer;
   
    
    dijkstrastate.DPtable.copyRowToRow(0, kIndexZeroLayerBackup);// backup zerolayer values

    for (int layer = 0; layer < dijkstrastate.numLayers + 1; layer++) {
        const int kOffsetPreviousLayer = ((layer+1) % 2) * dijkstrastate.maxDepthPerLayer;
        const int kOffsetCurrentLayer = (layer % 2) * dijkstrastate.maxDepthPerLayer;
        if (layer != 0) {
            dijkstrastate.DPtable.setBlockRowsConstant(kOffsetCurrentLayer, dijkstrastate.maxDepthPerLayer, CUSTOM_CUDA_INF);
            dijkstrastate.anyPathImproves.setConstant(false);
        }

        for (int k = 0; k < dijkstrastate.maxDepthPerLayer; k++) { // maxDepthPerLayer == 1 means no intralayer edges
            if (layer == 0 && k == 0) continue; // no previous layer available

            externalIter = 0;
            for (int vi = 0; vi < dijkstrastate.numNodesPerLayer; vi += externalIterOffset) {
                body_cuda<<<nBlocks, nThreads>>>(   dijkstrastate.DPtable, 
                                                    dijkstrastate.predecessor, 
                                                    dijkstrastate.anyPathImproves,
                                                    dijkstrastate.SRCIds,
                                                    dijkstrastate.costCuda,
                                                    dijkstrastate.currentRatio,
                                                    dijkstrastate.upperBound,
                                                    dijkstrastate.vertex2preceedingEdgeMapCuda,
                                                    layer,
                                                    dijkstrastate.maxDepthPerLayer,
                                                    dijkstrastate.numNodesPerLayer,
                                                    dijkstrastate.numLayers,
                                                    externalIter,
                                                    kOffsetPreviousLayer,
                                                    kOffsetCurrentLayer,
                                                    k);
                externalIter++;
            }
        } // k loop


        if (!(dijkstrastate.anyPathImproves.getMatrixValueFromHostAt(0)) && layer < dijkstrastate.numLayers && layer > 0) {
            if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "      > No path improved, early stopping" << std::endl;
            return false;
        }
    } // layer loop

    /*externalIter = 0;
    for (int vi = 0; vi < dijkstrastate.numNodesPerLayer; vi += externalIterOffset) {
        copyZeroLayer<<<nBlocks, nThreads>>>(   dijkstrastate.DPtable, 
                                                dijkstrastate.predecessor, 
                                                dijkstrastate.anyPathImproves,
                                                dijkstrastate.nodeBranchId, 
                                                dijkstrastate.branchId,
                                                dijkstrastate.maxDepthPerLayer,
                                                dijkstrastate.numNodesPerLayer,
                                                dijkstrastate.numLayers,
                                                externalIter,
                                                kIndexLastLayer, 
                                                kIndexZeroLayerBackup);
        externalIter++;   
    }*/

    /*
     Output part
     */
    const int kIndexLastLayer  = dijkstrastate.numLayers % 2; // this contains so far optimal values of duplicated zeroth layer
    bool anyPotentialNewLowerBound = false;
    bool* anyPotentialNewLowerBoundCuda;
    cudaMalloc((void **)&anyPotentialNewLowerBoundCuda,  1 * sizeof(bool));
    cudaMemcpy(anyPotentialNewLowerBoundCuda, &anyPotentialNewLowerBound, sizeof(bool), cudaMemcpyHostToDevice);
    externalIter = 0;
    for (int vi = 0; vi < dijkstrastate.numNodesPerLayer; vi += externalIterOffset) {

        extract_lower_bounds<<<nBlocks, nThreads>>>(dijkstrastate.DPtable,
                                                    dijkstrastate.nodeBranchIdCuda,
                                                    dijkstrastate.lbarray,
                                                    dijkstrastate.branchId,
                                                    dijkstrastate.maxDepthPerLayer,
                                                    dijkstrastate.numNodesPerLayer,
                                                    dijkstrastate.numLayers,
                                                    externalIter,
                                                    kIndexLastLayer,
                                                    anyPotentialNewLowerBoundCuda);
        externalIter++;   
    }
    cudaMemcpy(&anyPotentialNewLowerBound, anyPotentialNewLowerBoundCuda, sizeof(bool), cudaMemcpyDeviceToHost);
    if (!anyPotentialNewLowerBound) {
        if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "      > No non-inf lower bound" << std::endl;
        return false;
    }
    thrust::device_vector<LBTUPLE> lowerBoundsCuda(dijkstrastate.lbarray.data, dijkstrastate.lbarray.data+dijkstrastate.numNodesPerLayer);
    thrust::sort(lowerBoundsCuda.begin(), lowerBoundsCuda.end());
    dijkstrastate.lowerBounds = lowerBoundsCuda; // host vector


    bool firstIterPathSearchLoop = true;
    const int maxCycleLength = dijkstrastate.numLayers * dijkstrastate.maxDepthPerLayer;
    std::vector<int> cycle; cycle.reserve(maxCycleLength);
    dijkstrastate.upperboundFoundOnFirstTry = false;
    int* pred_cuda; cudaMalloc((void **)&pred_cuda,  sizeof(int));
    for (const LBTUPLE& lb : dijkstrastate.lowerBounds) {
        const float lowerBound = lb.lb;
        if (std::isinf(lowerBound)) {
            firstIterPathSearchLoop = false;
            continue;
        }
        const int v = lb.v;
        const int vOnLastLayer = v + dijkstrastate.numNodesPerLayer * dijkstrastate.numLayers * dijkstrastate.maxDepthPerLayer;
        
        extract_cycle<<<1, 1>>>(dijkstrastate.predecessor,
                                dijkstrastate.cycleCuda,
                                pred_cuda,
                                dijkstrastate.numNodesPerLayer,
                                dijkstrastate.maxDepthPerLayer,
                                maxCycleLength,
                                vOnLastLayer);
        int cycleBack;
        cudaMemcpy(&cycleBack, pred_cuda, sizeof(int), cudaMemcpyDeviceToHost);
        

        if (firstIterPathSearchLoop) {
            dijkstrastate.lowerBoundOfCurrentBranch = lowerBound;
        }
        if (v == cycleBack) { // the path starts and ends in the same vertex => lower bound is a valid path
            if (firstIterPathSearchLoop) {
                if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "    > found a valid cycle with lowest cost" << std::endl;
                dijkstrastate.upperboundFoundOnFirstTry = true;
            }
            const float newUpperBound = lowerBound;
            if (dijkstrastate.verbose) std::cout << dijkstrastate.prefix << "    > found valid cycle in branch cost=" << newUpperBound << std::endl;
            if (newUpperBound > dijkstrastate.upperBound && firstIterPathSearchLoop) {
                if (dijkstrastate.verbose) std::cout << dijkstrastate.prefix << "    > this branch cannot contain min mean (ub > best ub)" << std::endl;
                return false; // this branch cannot contain global optimum as best cycle is worse/equal compared to an already found cycle
            }
            if (newUpperBound < dijkstrastate.upperBound) {
                dijkstrastate.upperBound = newUpperBound;
                thrust::device_vector<int> cycleCudaThrust(dijkstrastate.cycleCuda.data, dijkstrastate.cycleCuda.data + dijkstrastate.cycleCuda.size());
                dijkstrastate.cycle = cycleCudaThrust;
            }
            break;
        }
        else if (firstIterPathSearchLoop) {
            if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "    > best lower bound in branch " << lowerBound << std::endl;
            dijkstrastate.branchingSrcNode = v;
            dijkstrastate.branchingTargetNode = cycleBack;
        }
        firstIterPathSearchLoop = false; // only first iteration ;)
    }

    return true;
}




} //  namespace cuda
} //  namespace dijkstra