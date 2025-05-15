#include "ctr_bnb_subroutine.cuh"
#include <algorithm>

namespace ctrsolver {
namespace cuda {

// see https://stackoverflow.com/questions/18963293/cuda-atomics-change-flag/18968893#18968893
__device__ void acquire_semaphore(volatile int *lock) {
    while (atomicCAS((int *)lock, 0, 1) != 0);
}
__device__ void release_semaphore(volatile int *lock) {
    *lock = 0;
    __threadfence_block();
}
int getOriginalGraphIndex(const CTRGPUSTATE& ctrstate, const int pred) {
    const int numNodesPerLayerWithHelperLayers = ctrstate.numNodesPerLayer * ctrstate.maxDepthPerLayer;
    const int cycleLayer = pred / numNodesPerLayerWithHelperLayers;
    const int predInHelperLayer = (pred - cycleLayer * numNodesPerLayerWithHelperLayers);
    const int cycleHelperLayer = predInHelperLayer / ctrstate.numNodesPerLayer ;
    const int predInNormalLayer = predInHelperLayer - cycleHelperLayer * ctrstate.numNodesPerLayer;
    const int indexInOriginalGraph = predInNormalLayer + cycleLayer * ctrstate.numNodesPerLayer;
    if (DEBUG_CRT_CYCLE_SOLVER && indexInOriginalGraph < 0 )
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
    if (DEBUG_CRT_CYCLE_SOLVER && indexInOriginalGraph < 0 )
        printf("indexInOriginalGraph < 0");
    return indexInOriginalGraph;
}

std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> computeRatioForCycle(const CTRGPUSTATE& ctrstate,
                                                                               const std::vector<int> cycle,
                                                                               const Eigen::MatrixXi& productspace,
                                                                               const Eigen::MatrixXi& SRCIds,
                                                                               const Eigen::MatrixXi& TRGTIds,
                                                                               const bool writeOutput) {
    const int cycleLength = cycle.size();
    PRECISION_CTR costSum = 0;
    PRECISION_CTR timeSum = 0;
    PRECISION_CTR wSum = 0;
    int cycleElementsAdded = 0;
    Eigen::MatrixXi outputEdges;
    if (writeOutput) {
        outputEdges = Eigen::MatrixXi(cycleLength+1, productspace.cols()); outputEdges.setConstant(-1);
    }
    for (int i = 0; i < cycleLength-1; i++) {
        const int cycleSrcIdx = i;
        const int cycleTrgtIdx = i+1;
        const int srcIdx = cycle.at(cycleSrcIdx);
        int trgtIdx = cycle.at(cycleTrgtIdx);
        if (trgtIdx < ctrstate.numNodesPerLayer && srcIdx > ctrstate.numNodesPerLayer) {
            trgtIdx += ctrstate.numNodes;
        }
        int edgeIdx = -2;
        if (DEBUG_CRT_CYCLE_SOLVER && trgtIdx >= ctrstate.vertex2preceedingEdgeMap.size()) {
            std::cout << "trgtIdx >= ctrstate.vertex2preceedingEdgeMap.size()" << std::endl;
        }
        for (const auto& it : ctrstate.vertex2preceedingEdgeMap.at(trgtIdx)) {
            const long e = it;
            if (e != -1 && SRCIds(e) == srcIdx) {
                edgeIdx = e;
                break;
            }
            if (e == -1) {
                continue;
            }
        }
        //std::cout << ctrstate.cost(edgeIdx) << " " << ctrstate.time(edgeIdx) << std::endl;
        costSum += ctrstate.cost(edgeIdx);
        timeSum += ctrstate.time(edgeIdx);
        wSum += ctrstate.cost(edgeIdx) - ctrstate.currentRatio * ctrstate.time(edgeIdx);
        if (writeOutput) outputEdges.row(cycleElementsAdded) = productspace.row(edgeIdx);
        cycleElementsAdded++;
    }
    if (writeOutput) outputEdges.conservativeResize(cycleElementsAdded, productspace.cols());
    const PRECISION_CTR costTimeRatio = costSum / timeSum;
    return std::make_tuple(costTimeRatio, wSum, outputEdges);
}
std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> computeRatioForCurrentCycle(CTRGPUSTATE& ctrstate,
                                                                                      const Eigen::MatrixXi& productspace,
                                                                                      const Eigen::MatrixXi& SRCIds,
                                                                                      const Eigen::MatrixXi& TRGTIds,
                                                                                      const bool writeOutput) {
    return computeRatioForCycle(ctrstate, ctrstate.negativeCycle, productspace, SRCIds, TRGTIds, writeOutput);
}

__global__ void setup_dp_table_cuda(CudaMatrix<PRECISION_CTR> DPTABLE,
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


__global__ void body_cuda(CudaMatrix<PRECISION_CTR> DPTABLE, 
                          CudaMatrix<int> PREDECESSOR, 
                          CudaMatrix<bool> anyPathImproves,
                          const CudaMatrix<int> SRCIds,
                          const CudaMatrix<PRECISION_CTR> cost,
                          const CudaMatrix<PRECISION_CTR> time,
                          const PRECISION_CTR currentRatio,
                          const PRECISION_CTR upperBound,
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
        PRECISION_CTR newCost = cost(e) - currentRatio * time(e);
        long srcIdx = SRCIds(e);
        const bool isIntraLayerEdge = (srcIdx / numNodesPerLayer) == layer;

        if (isIntraLayerEdge && k != 0) {
            if (maxDepthPerLayer <= 1) continue;
            const int srcIdxInLayer = srcIdx - layer * numNodesPerLayer;
            if (custom_is_inf(DPTABLE(kOffsetCurrentLayer + k - 1, srcIdxInLayer))) continue;

            const PRECISION_CTR newVal = DPTABLE(kOffsetCurrentLayer + k - 1, srcIdxInLayer) + newCost;
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

                const PRECISION_CTR newVal = DPTABLE(kOffsetPreviousLayer + kk, srcIdxInLayer) + newCost;
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
                              const int vStartIndexOnFirstOrLastLayer,
                              const int numNodesPerLayer,
                              const int maxDepthPerLayer,
                              const int maxLength,
                              const int vStartIndexOnFirstLayer) {
    cycle.clear();
    int pred = PREDECESSOR(vStartIndexOnFirstOrLastLayer);
    if (vStartIndexOnFirstOrLastLayer != vStartIndexOnFirstLayer) {
        cycle.push_back(getOriginalGraphIndexCuda(numNodesPerLayer, maxDepthPerLayer, vStartIndexOnFirstOrLastLayer));
    }
    cycle.push_back(getOriginalGraphIndexCuda(numNodesPerLayer, maxDepthPerLayer, pred));
    for (int kk = maxLength - 1; kk > 0; kk--) {
        if (pred == -1) break;
        pred = PREDECESSOR(pred);

        cycle.push_back(getOriginalGraphIndexCuda(numNodesPerLayer, maxDepthPerLayer, pred));
        if (pred == vStartIndexOnFirstLayer) break;
    }
}

__global__ void copyZeroLayer(  CudaMatrix<PRECISION_CTR> DPTABLE, 
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


__global__ void check_for_cycles(   const CudaMatrix<PRECISION_CTR> DPTABLE, 
                                    const CudaMatrix<int> PREDECESSOR, 
                                    const CudaMatrix<int> nodeBranchId,
                                    CudaMatrix<bool> vertexInCycleEncounter,
                                    CudaMatrix<int> cycleDetectionHelper,
                                    const int branchId,
                                    const long numNodesPerLayer,
                                    const int externalIter,
                                    const PRECISION_CTR tolerance,
                                    bool* existsNegativeCycle,
                                    const bool searchForZeroCycle,
                                    const int maxCycleLength) {
    const int blockIndex = blockIdx.x;
    const int threadIndex = threadIdx.x + blockIndex * blockDim.x + externalIter * (blockDim.x * gridDim.x);
    if (threadIndex >= numNodesPerLayer) return;
    const int vi = threadIndex;
    const PRECISION_CTR lowerBound = DPTABLE(0, vi);
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
BELLMANFORDSTATUS mooreBellmanFordSubroutineGPU(CTRGPUSTATE& ctrstate,
                                                const Eigen::MatrixXi& productspace,
                                                const Eigen::MatrixXi& SRCIds,
                                                const Eigen::MatrixXi& TRGTIds,
                                                const bool searchForZeroCycle,
                                                const bool searchForUpperBound) {
    const int externalIterOffset = ctrstate.numThreadBlocks * ctrstate.threadsPerBlock;
    const int nBlocks = ctrstate.numThreadBlocks;
    const int nThreads = ctrstate.threadsPerBlock;
    const int numThreads = 1;
    // setup
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    Eigen::MatrixXf perVertexLowerBounds(ctrstate.numNodesPerLayer, numThreads);
    perVertexLowerBounds.setConstant(CUSTOM_CUDA_INF);
    Eigen::MatrixXi perVertexLbK(ctrstate.numNodesPerLayer, numThreads);
    ctrstate.DPtable.setConstant(CUSTOM_CUDA_INF);

    int indexNodeInLayer = -1;
    int numNodesInBranch = 0;
    int* num_nodes_in_branch_device;
    int* index_node_in_layer_device;
    cudaMalloc((void **)&num_nodes_in_branch_device,  1 * sizeof(int));
    cudaMalloc((void **)&index_node_in_layer_device,  1 * sizeof(int));
    cudaMemcpy(num_nodes_in_branch_device, &numNodesInBranch, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(index_node_in_layer_device, &indexNodeInLayer, sizeof(int), cudaMemcpyHostToDevice);
    int externalIter = 0;
    for (int i = 0; i < ctrstate.nodeBranchId.rows(); i += externalIterOffset) {
        setup_dp_table_cuda<<<nBlocks, nThreads>>>( ctrstate.DPtable, 
                                                    ctrstate.predecessor, 
                                                    ctrstate.nodeBranchIdCuda, 
                                                    num_nodes_in_branch_device,
                                                    index_node_in_layer_device,
                                                    externalIter,
                                                    ctrstate.branchId,
                                                    ctrstate.numNodes,
                                                    ctrstate.numNodesPerLayer);
        externalIter++;
    }
    if (DEBUG_CTR_CUDA) checkCudaError("setup end: ");
    cudaMemcpy(&numNodesInBranch, num_nodes_in_branch_device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&indexNodeInLayer, index_node_in_layer_device, sizeof(int), cudaMemcpyDeviceToHost);
    if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Num nodes in branch " << numNodesInBranch << std::endl;
    ctrstate.numNodesInBranch = numNodesInBranch;
    if (numNodesInBranch == 0) {
        std::cout << "NO NODES IN BRANCH, this should not happen" << std::endl; exit(-1);
    }


    // dynamic programming loop: sweep through all layers
    bool negativeCycleFound = false;
    int iter = 0;
    const int kIndexZeroLayerBackup  = 2 * ctrstate.maxDepthPerLayer;
   
    while (!negativeCycleFound) { // this should terminate if we found a negative cycle or if no negative cycle exists or after max iters == num nodes // backup zerolayer values
        ctrstate.DPtable.copyRowToRow(0, kIndexZeroLayerBackup);// backup zerolayer values

        for (int layer = 0; layer < ctrstate.numLayers + 1; layer++) {
            const int kOffsetPreviousLayer = ((layer+1) % 2) * ctrstate.maxDepthPerLayer;
            const int kOffsetCurrentLayer = (layer % 2) * ctrstate.maxDepthPerLayer;
            if (layer != 0) {
                ctrstate.DPtable.setBlockRowsConstant(kOffsetCurrentLayer, ctrstate.maxDepthPerLayer, CUSTOM_CUDA_INF);
                ctrstate.anyPathImproves.setConstant(false);
            }

            for (int k = 0; k < ctrstate.maxDepthPerLayer; k++) { // maxDepthPerLayer == 1 means no intralayer edges
                if (layer == 0 && k == 0) continue; // no previous layer available

                externalIter = 0;
                for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi += externalIterOffset) {
                    body_cuda<<<nBlocks, nThreads>>>(   ctrstate.DPtable, 
                                                        ctrstate.predecessor, 
                                                        ctrstate.anyPathImproves,
                                                        ctrstate.SRCIds,
                                                        ctrstate.costCuda,
                                                        ctrstate.timeCuda,
                                                        ctrstate.currentRatio,
                                                        ctrstate.upperBound,
                                                        ctrstate.vertex2preceedingEdgeMapCuda,
                                                        layer,
                                                        ctrstate.maxDepthPerLayer,
                                                        ctrstate.numNodesPerLayer,
                                                        ctrstate.numLayers,
                                                        externalIter,
                                                        kOffsetPreviousLayer,
                                                        kOffsetCurrentLayer,
                                                        k);
                    externalIter++;
                }
            } // k loop


            if (!(ctrstate.anyPathImproves.getMatrixValueFromHostAt(0)) && layer < ctrstate.numLayers && layer > 0) {
                if (searchForZeroCycle) std::cout << "TODO: what if we search for zero cycle and this happens:" << std::endl;
                if (ctrstate.verbose)  std::cout << ctrstate.prefix << "      > No path improved => no negative cycle exists" << std::endl;
                const auto outOld = computeRatioForCycle(ctrstate, ctrstate.negativeCycle, productspace, SRCIds, TRGTIds);
                const int integralCycleCostOld = std::round(std::get<1>(outOld));
                if (integralCycleCostOld == 0) {
                    if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Zero cycle found from previous iteration with length: " << ctrstate.negativeCycle.size() << std::endl;
                    return BELLMANFORDSTATUS::ZERO_CYCLE;
                }
                return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
            }
        } // layer loop

        if (searchForUpperBound) {
            // search for the lowest cost path from indexNodeInLayer to indexNodeInLayer (upperbound search should invoke with single index in branch)
            const int maxLength = ctrstate.numLayers * ctrstate.maxDepthPerLayer;
            const int firstV = indexNodeInLayer + ctrstate.numNodesPerLayer * ctrstate.numLayers * ctrstate.maxDepthPerLayer;
            if (maxLength > 100 * ctrstate.numLayers * ctrstate.maxDepthPerLayer) {
                std::cout << "Cycle is too long to fit into reserved memory, this very likely will lead to undefined behaviour" << std::endl;
            }
            extract_cycle<<<1, 1>>>(ctrstate.predecessor,
                                    ctrstate.cycleCuda, 
                                    firstV, 
                                    ctrstate.numNodesPerLayer,
                                    ctrstate.maxDepthPerLayer,
                                    maxLength,
                                    indexNodeInLayer);

            const thrust::device_vector<int> cycleCudaThrust(ctrstate.cycleCuda.data, ctrstate.cycleCuda.data + ctrstate.cycleCuda.size());
            const thrust::host_vector<int> cycleHostThrust = cycleCudaThrust;
            std::vector<int> cycle(cycleHostThrust.begin(), cycleHostThrust.end());
            std::reverse(cycle.begin(), cycle.end());
            //cycle.push_back(cycle.front() + ctrstate.numNodes);

            ctrstate.negativeCycle = cycle;
            const auto outOld = computeRatioForCycle(ctrstate, cycle, productspace, SRCIds, TRGTIds);
            const int integralCycleCostOld = std::round(std::get<1>(outOld));
            if (integralCycleCostOld == 0) {
                return BELLMANFORDSTATUS::ZERO_CYCLE;
            }
            if (integralCycleCostOld < 0) {
                return BELLMANFORDSTATUS::FOUND_NEGATIVE_CYCLE;
            }
            return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
        }

        // check if layer zero has improvement and rewrite zero layer in DP table
        const int kIndexLastLayer  = ctrstate.numLayers % 2; // this contains so far optimal values of duplicated zeroth layer
        ctrstate.anyPathImproves.setConstant(false);

        externalIter = 0;
        for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi += externalIterOffset) {
            copyZeroLayer<<<nBlocks, nThreads>>>(   ctrstate.DPtable, 
                                                    ctrstate.predecessor, 
                                                    ctrstate.anyPathImproves,
                                                    ctrstate.nodeBranchId, 
                                                    ctrstate.branchId,
                                                    ctrstate.maxDepthPerLayer,
                                                    ctrstate.numNodesPerLayer,
                                                    ctrstate.numLayers,
                                                    externalIter,
                                                    kIndexLastLayer, 
                                                    kIndexZeroLayerBackup);
            externalIter++;   
        }

        if (!(ctrstate.anyPathImproves.getMatrixValueFromHostAt(0))) {
            if (ctrstate.verbose)  std::cout << ctrstate.prefix << "      > No path improved => no negative cycle exists" << std::endl;
            const auto outOld = computeRatioForCycle(ctrstate, ctrstate.negativeCycle, productspace, SRCIds, TRGTIds);
            const int integralCycleCostOld = std::round(std::get<1>(outOld));
            if (integralCycleCostOld == 0) {
                if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Zero cycle found from previous iteration with length: " << ctrstate.negativeCycle.size() << std::endl;
                return BELLMANFORDSTATUS::ZERO_CYCLE;
            }
            return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
        }


        // check for negative cycle
        int vOnFirstLayerInCycle = -1, cycleLength = -1;
        const int maxCycleLength = ctrstate.numNodes;

        externalIter = 0;
        bool existsNegativeCycle = false;
        bool existsNegativeCycleLocal = false;
        bool* exists_negative_cycle_device;
        cudaMalloc((void **)&exists_negative_cycle_device,  1 * sizeof(bool));
        cudaMemcpy(exists_negative_cycle_device, &existsNegativeCycle, sizeof(bool), cudaMemcpyHostToDevice);

        for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi += THREAD_VERTEX_CYCLE_ENCOUNTER) {
            ctrstate.cycleDetectionHelper.setConstant(-1);
            ctrstate.vertexInCycleEncounter.setConstant(false);
            check_for_cycles<<<1, THREAD_VERTEX_CYCLE_ENCOUNTER>>>( ctrstate.DPtable, 
                                                                    ctrstate.predecessor, 
                                                                    ctrstate.nodeBranchId,
                                                                    ctrstate.vertexInCycleEncounter,
                                                                    ctrstate.cycleDetectionHelper,
                                                                    ctrstate.branchId,
                                                                    ctrstate.numNodesPerLayer,
                                                                    externalIter,
                                                                    ctrstate.tolerance,
                                                                    exists_negative_cycle_device,
                                                                    searchForZeroCycle,
                                                                    maxCycleLength);

            cudaMemcpy(&existsNegativeCycleLocal, exists_negative_cycle_device, sizeof(bool), cudaMemcpyDeviceToHost);
            
            if (existsNegativeCycleLocal) {
                existsNegativeCycleLocal = false;
                cudaMemcpy(exists_negative_cycle_device, &existsNegativeCycleLocal, sizeof(bool), cudaMemcpyHostToDevice);
                Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> indexAndLength;
                ctrstate.cycleDetectionHelper.copyToEigen(indexAndLength);

                PRECISION_CTR bestLb = CUSTOM_CUDA_INF;
                for (int vii = 0; vii < std::min(ctrstate.numNodesPerLayer, (long)THREAD_VERTEX_CYCLE_ENCOUNTER); vii++) {
                    const int vCandidate = indexAndLength(vii, 0);
                    if (vCandidate == -1) continue;
                    const int cycleLengthCandidate = indexAndLength(vii, 1);
                    const PRECISION_CTR lbCandidate = indexAndLength(vii, 2);
                    if (lbCandidate < bestLb) {
                        existsNegativeCycle = true;
                        vOnFirstLayerInCycle = vCandidate;
                        cycleLength = cycleLengthCandidate;
                    }
                }
                //break;
            }
            externalIter++;
        }

        // extract negative cycle if one is found
        if (existsNegativeCycle) {
            negativeCycleFound = true;
            if (cycleLength > 100 * ctrstate.numLayers * ctrstate.maxDepthPerLayer) {
                std::cout << "Cycle is too long to fit into reserved memory, this very likely will lead to undefined behaviour" << std::endl;
            }
            extract_cycle<<<1, 1>>>(ctrstate.predecessor,
                                    ctrstate.cycleCuda, 
                                    vOnFirstLayerInCycle, 
                                    ctrstate.numNodesPerLayer,
                                    ctrstate.maxDepthPerLayer,
                                    cycleLength,
                                    vOnFirstLayerInCycle);

            const thrust::device_vector<int> cycleCudaThrust(ctrstate.cycleCuda.data, ctrstate.cycleCuda.data + ctrstate.cycleCuda.size());
            const thrust::host_vector<int> cycleHostThrust = cycleCudaThrust;
            std::vector<int> cycle(cycleHostThrust.begin(), cycleHostThrust.end());

            std::reverse(cycle.begin(), cycle.end());
            cycle.push_back(cycle.front() + ctrstate.numNodes);

            const auto out = computeRatioForCycle(ctrstate, cycle, productspace, SRCIds, TRGTIds);
            const int integralCycleCost = std::round(std::get<1>(out));
            if (integralCycleCost < 0) {
                ctrstate.negativeCycle = cycle;
                if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Negative cycle with length: " << cycle.size() << std::endl;
                return BELLMANFORDSTATUS::FOUND_NEGATIVE_CYCLE;
            }
            else if (integralCycleCost == 0) {
                continue; /* we should not stop if we have found a zero cycle bc this could be just the previous cycle
                ctrstate.negativeCycle = cycle;
                if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Zero cycle found with length: " << cycle.size() << std::endl;
                return BELLMANFORDSTATUS::ZERO_CYCLE;*/
            }
            else {
                // we should actually never end up here 
                std::cout << ctrstate.prefix << "ERROR: We should never have integral cost > 0 if we have found negative cycle" << std::endl;
                const auto outOld = computeRatioForCycle(ctrstate, ctrstate.negativeCycle, productspace, SRCIds, TRGTIds);
                const int integralCycleCostOld = std::round(std::get<1>(outOld));
                if (integralCycleCostOld == 0) {
                    if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Zero cycle found from previous iteration with length: " << ctrstate.negativeCycle.size() << std::endl;
                    return BELLMANFORDSTATUS::ZERO_CYCLE;
                }
                std::cout << "cycle length = " << ctrstate.negativeCycle.size() << std::endl;;
                if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > No negative cycle found, cost = " << integralCycleCost << std::endl;
                return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
            }
        }


        iter++;
        if (iter >= ctrstate.numNodes-1) {
            if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Maximum simple path length reached and no negative cycle found => no negative cycle exists" << std::endl;
            return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
        }
    } // while(!negativeCycleFound)*/
    return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
}




bool lawlerSoubroutine(CTRGPUSTATE& ctrstate,
                       const Eigen::MatrixXi& productspace,
                       const Eigen::MatrixXi& SRCIds,
                       const Eigen::MatrixXi& TRGTIds) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int i = 0;
    const int maxiter = 1000;
    bool existsNegativeCycle = true;
    if (ctrstate.verbose) std::cout << ctrstate.prefix << "     + Running Lawlers alogrithm on GPU.. maxiter = " << maxiter << std::endl;
    PRECISION_CTR oldRatio = ctrstate.currentRatio;
    while (existsNegativeCycle) {
        if (ctrstate.verbose) std::cout << ctrstate.prefix << "     + Lawler iter = " << i+1 << ", current ratio = " << ctrstate.currentRatio << std::endl;
        const BELLMANFORDSTATUS bfstatus = mooreBellmanFordSubroutineGPU(ctrstate, productspace, SRCIds, TRGTIds, false);
        existsNegativeCycle = !(bfstatus == NO_NEGATIVE_CYCLE);

        std::chrono::steady_clock::time_point titer = std::chrono::steady_clock::now();
        if (bfstatus == ZERO_CYCLE) {
            if (ctrstate.verbose) std::cout << ctrstate.prefix << "     +> Found zero cycle => optimum (" << std::chrono::duration_cast<std::chrono::milliseconds>(titer - t1).count() << "  [ms])" << std::endl;
            break;
        }
        if (bfstatus == NO_NEGATIVE_CYCLE) {
            if (ctrstate.verbose) std::cout << ctrstate.prefix << "     +> Did not find negative cycle (" << std::chrono::duration_cast<std::chrono::milliseconds>(titer - t1).count() << "  [ms])" << std::endl;
            break;
        }


        const std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> ratioAndWeight = computeRatioForCurrentCycle(ctrstate, productspace, SRCIds, TRGTIds);
        if (ctrstate.verbose) std::cout << ctrstate.prefix << "     +> Found negative cycle (" << std::chrono::duration_cast<std::chrono::milliseconds>(titer - t1).count() << "  [ms])" << std::endl;

        ctrstate.currentRatio = std::get<0>(ratioAndWeight);
        if (std::abs(oldRatio - ctrstate.currentRatio) < ctrstate.tolerance) {
            std::cout << oldRatio << " "<< ctrstate.currentRatio << std::endl;
            if (ctrstate.verbose) std::cout << ctrstate.prefix << "     +> Ratio did not change, assuming optimum found (" << std::chrono::duration_cast<std::chrono::milliseconds>(titer - t1).count() << "  [ms])" << std::endl;
            break;
        }
        if (ctrstate.currentRatio -  ctrstate.tolerance > oldRatio) {
            if (ctrstate.verbose) std::cout << ctrstate.prefix << "     +> Ratio increased, stopping (" << std::chrono::duration_cast<std::chrono::milliseconds>(titer - t1).count() << "  [ms])" << std::endl;
            break;
        }
        oldRatio = ctrstate.currentRatio;

        if (i > maxiter) {
            std::cout << ctrstate.prefix << "     + Reached maxiter, stopping (maxiter = " << maxiter << ")" << std::endl;
            break;
        }
        i++;
    }

    ctrstate.lowerBoundOfCurrentBranch = ctrstate.currentRatio;

    // check if cycle wraps around multiple times
    const int startVertexOfCycle = ctrstate.negativeCycle.front();
    int numWraps = 0;
    ctrstate.zeroLayerHits.clear();
    bool enteredFirstLayer = true;
    bool leftFirstLayer = false;
    for (const auto& it : ctrstate.negativeCycle) {
        const int vertex = it;
        if (leftFirstLayer && vertex < ctrstate.numNodesPerLayer) {
            enteredFirstLayer = true;
            leftFirstLayer = false;
        }

        if (enteredFirstLayer) {
            ctrstate.zeroLayerHits.push_back(vertex);
            enteredFirstLayer = false;
        }

        if (vertex >= ctrstate.numNodesPerLayer && !leftFirstLayer) {
            numWraps++;
            leftFirstLayer = true;
        }

    }
    const bool cycleWrapsAroundMultipleTimes = numWraps > 1;
    if (ctrstate.verbose) std::cout << ctrstate.prefix << "   + Cycle wraps around " << numWraps << " times" << std::endl;
    return cycleWrapsAroundMultipleTimes;
}


} //  namespace ctrsolver
} //  namespace cuda