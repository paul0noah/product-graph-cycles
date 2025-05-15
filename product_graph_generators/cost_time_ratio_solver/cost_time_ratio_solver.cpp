//
//  mean_cycle_solver.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 07.10.24
//
#include "cost_time_ratio_solver.hpp"

#include <chrono>
#include <string.h>
#include <helper/utils.hpp>
#include <type_traits>
#include <tsl/robin_map.h>
#include "helper/minheap.hpp"


namespace ctrsolver {
#if defined(_OPENMP)
int omp_thread_count() {
    // workaround for omp_get_num_threads()
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}
inline int getThreadId() {
    return omp_get_thread_num();
}
#else
int omp_thread_count() {
    return 1;
}
inline int getThreadId() {
    return 0;
}
#endif

bool smaller(const LBTUPLE& a, LBTUPLE& b) {
    return std::get<0>(a) < std::get<0>(b);
}

std::vector<tsl::robin_set<long>> CostTimeRatioSolver::buildingInGraph(const long numTotalNodes,
                                                                       const long numNodes) {
    std::vector<tsl::robin_set<long>> vertex2preceedingEdgeMap;  vertex2preceedingEdgeMap.reserve(numTotalNodes);
    for (int v = 0; v < numTotalNodes; v++) {
        vertex2preceedingEdgeMap.push_back(tsl::robin_set<long>());
        if (v < numNodesPerLayer) {
            vertex2preceedingEdgeMap.at(v).insert(-1); // used for dummy edge cost
        }
    }
    for (int e = 0; e < TRGTIds.rows(); e++) {
        const int v = TRGTIds(e);

        // handle dummy edges: every node on the 0-th layer is connected to respective dummy vertex
        // while respective dummy vertex is connected to nodes from which the original connection came from
        if (v < numNodesPerLayer && SRCIds(e) > numNodesPerLayer)  {
            const long dummyV = numNodes + v;
            vertex2preceedingEdgeMap.at(dummyV).insert(e);
            continue;
        }

        // all normal edges
        vertex2preceedingEdgeMap.at(v).insert(e);
    }
    return vertex2preceedingEdgeMap;
}

CostTimeRatioSolver::CostTimeRatioSolver(const Eigen::MatrixXi& productspace,
                                 Eigen::MatrixXd& cost,
                                 const Eigen::MatrixXi& SRCIds,
                                 const Eigen::MatrixXi& TRGTIds,
                                 const long numNodesPerLayer,
                                 const int imaxDepth) : productspace(productspace), cost(cost), SRCIds(SRCIds), TRGTIds(TRGTIds), numNodesPerLayer(numNodesPerLayer) {
    verbose = true;
    prefix = "[CTRSolver] ";
    branchGraph = std::vector<tsl::robin_set<long>>();
    cycleChecking = true;
    maxDepth = imaxDepth;
    tolerance = 1e-8;
    if(verbose) std::cout << prefix << "Running CRTSolver with tolerance = " << tolerance << " and maxDepthPerLayer = " << maxDepth << std::endl;

}

CostTimeRatioSolver::~CostTimeRatioSolver() {

}

std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> computeRatioForCycle(const CTRCPUSTATE& ctrstate,
                                                            const std::vector<int> cycle,
                                                            const Eigen::MatrixXi& productspace,
                                                            const Eigen::MatrixXi& SRCIds,
                                                            const Eigen::MatrixXi& TRGTIds,
                                                            const bool writeOutput=false) {
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
        //std::cout << srcIdx << " " << trgtIdx << std::endl;
        if (trgtIdx < ctrstate.numNodesPerLayer && srcIdx > ctrstate.numNodesPerLayer) {
            trgtIdx += ctrstate.numNodes;
        }
        int edgeIdx = -2;
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
std::tuple<PRECISION_CTR, PRECISION_CTR, Eigen::MatrixXi> computeRatioForCurrentCycle(CTRCPUSTATE& ctrstate,
                                                     const Eigen::MatrixXi& productspace,
                                                     const Eigen::MatrixXi& SRCIds,
                                                     const Eigen::MatrixXi& TRGTIds,
                                                     const bool writeOutput=false) {
    return computeRatioForCycle(ctrstate, ctrstate.negativeCycle, productspace, SRCIds, TRGTIds, writeOutput);
}

void branch(CTRCPUSTATE& ctrstate,
            const Eigen::MatrixXi& productspace,
            const Eigen::MatrixXd& cost,
            const Eigen::MatrixXi& SRCIds,
            const Eigen::MatrixXi& TRGTIds) {
    const bool branchGraphNeeded = ctrstate.branchGraph.size() > 0;
    const auto& branchGraph = branchGraphNeeded ? ctrstate.branchGraph : ctrstate.vertex2preceedingEdgeMap;

    MinHeap heap(ctrstate.numNodesPerLayer);
    for (int i = 0; i < ctrstate.numNodesPerLayer; i++) {
        ctrstate.DPtable(0, i) = -1;
        if (i == ctrstate.branchingSrcNode || i == ctrstate.branchingTargetNode) {
            ctrstate.DPtable(0, i) = 1;
            continue;
        }
        if (ctrstate.nodeBranchId(i) == ctrstate.branchId) {
            heap.push(std::numeric_limits<double>::infinity(), i);
            ctrstate.DPtable(0, i) = -1; // node unprocessed
        }
    }
    int branchOffset = 0;
    for (const auto& it : ctrstate.zeroLayerHits) {
        const int indexOfZeroLayerHitVertex = it;
        heap.push(0.0, indexOfZeroLayerHitVertex);
        if (branchOffset > 0) { // one vertex is kept in current branch
            ctrstate.nodeBranchId(indexOfZeroLayerHitVertex) = ctrstate.maxBranchId + branchOffset;
        }
        branchOffset++;
    }

    int counter = 0;
    for (int v = 0; v < ctrstate.numNodesPerLayer; v++) {
        if (ctrstate.nodeBranchId(v) == ctrstate.branchId) {
            ctrstate.nodeBranchId(v) = ctrstate.maxBranchId + (int)( (counter / ctrstate.numNodesInBranch) * branchOffset);
        }
    }
    ctrstate.maxBranchId += branchOffset;
    return;

    counter = 0;
    for (const LBTUPLE& lb : ctrstate.lowerBounds) { // lower bounds sorted by their cost
        const int v = std::get<1>(lb) - ctrstate.numNodesPerLayer * ctrstate.maxDepthPerLayer * ctrstate.numLayers;
        if (v < 0 || v >= ctrstate.numNodesPerLayer) continue;
        bool vInZeroLayerHits = false;
        for (const auto& it : ctrstate.zeroLayerHits) {
            if (v == it) vInZeroLayerHits = true;
        }
        if (vInZeroLayerHits) continue;
        if (ctrstate.nodeBranchId(v) != ctrstate.branchId) continue;

        // basically put the vertices with worst lower bound to different branch
        if (counter > std::max(1, ctrstate.numNodesInBranch / branchOffset)) {
            heap.push(0.1, v);
            ctrstate.nodeBranchId(v) = ctrstate.maxBranchId + (int)(counter / ctrstate.numNodesInBranch * branchOffset);
        }
        counter++;
    }


    while(!heap.isEmpty()) {
        const std::pair<double, long> current = heap.pop();
        const int currentNodeId = current.second;
        const float currentNodeCost = current.first;
        ctrstate.DPtable(0, currentNodeId) = 1.0; // node is processed

        for (const auto& it : branchGraph.at(currentNodeId)) {
            const long e = it;
            if (e == -1) continue; // edge coming from previous layer
            const int srcId = SRCIds(e);
            if (ctrstate.DPtable(0, srcId) < 0) { // srcId node not yet finished processing
                const float oldValue = heap.peakKey(srcId);
                const float newValue = currentNodeCost + 1.0;
                if (newValue < oldValue) {
                    heap.decrease(srcId, newValue);
                    ctrstate.nodeBranchId(srcId) = ctrstate.nodeBranchId(currentNodeId);
                }
            }
        }
    }

}


/*






 Moore-Bellman-Ford subroutine






 */
int getOriginalGraphIndex(const CTRCPUSTATE& ctrstate, const int pred) {
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
BELLMANFORDSTATUS mooreBellmanFordSubroutineCPU(CTRCPUSTATE& ctrstate,
                                const Eigen::MatrixXi& productspace,
                                const Eigen::MatrixXi& SRCIds,
                                const Eigen::MatrixXi& TRGTIds,
                                const bool searchForZeroCycle=false,
                                const bool searchForUpperBound=false) {
    #if defined(_OPENMP)
    const int numThreads = omp_thread_count();
    #else
    const int numThreads = 1;
    #endif
    // setup
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    Eigen::MatrixXf perVertexLowerBounds(ctrstate.numNodesPerLayer, numThreads);
    perVertexLowerBounds.setConstant(std::numeric_limits<float>::infinity());
    Eigen::MatrixXi perVertexLbK(ctrstate.numNodesPerLayer, numThreads);
    ctrstate.DPtable.setConstant(std::numeric_limits<float>::infinity());

    int indexNodeInLayer = -1;
    int numNodesInBranch = 0;
    #if defined(_OPENMP)
    #pragma omp parallel for reduction (+:numNodesInBranch)
    #endif
    for (int i = 0; i < ctrstate.nodeBranchId.rows(); i++) {
        const int startVertexId = i;
        if (ctrstate.nodeBranchId(startVertexId) == ctrstate.branchId) {
            indexNodeInLayer = startVertexId;
            numNodesInBranch++;
            ctrstate.DPtable(0, startVertexId) = 0;
            ctrstate.predecessor(startVertexId) = -1;
        }
    }
    if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Num nodes in branch " << numNodesInBranch << std::endl;
    ctrstate.numNodesInBranch = numNodesInBranch;
    if (numNodesInBranch == 0) {
        std::cout << "NO NODES IN BRANCH, this should not happen" << std::endl; exit(-1);
    }

    // dynamic programming loop: sweep through all layers
    Eigen::MatrixX<bool> anyPathImproves(numThreads, 1); anyPathImproves.setConstant(true);
    bool negativeCycleFound = false;
    int iter = 0;
    const int kIndexZeroLayerBackup  = 2 * ctrstate.maxDepthPerLayer;
    while (!negativeCycleFound) { // this should terminate if we found a negative cycle or if no negative cycle exists or after max iters == num nodes
        ctrstate.DPtable.row(kIndexZeroLayerBackup) = ctrstate.DPtable.row(0); // backup zerolayer values


        for (int layer = 0; layer < ctrstate.numLayers + 1; layer++) {
            const int kOffsetPreviousLayer = ((layer+1) % 2) * ctrstate.maxDepthPerLayer;
            const int kOffsetCurrentLayer = (layer % 2) * ctrstate.maxDepthPerLayer;
            if (layer != 0) {
                ctrstate.DPtable.block(kOffsetCurrentLayer, 0, ctrstate.maxDepthPerLayer, ctrstate.numNodesPerLayer).setConstant(std::numeric_limits<float>::infinity());
                anyPathImproves.setConstant(false);
            }

            for (int k = 0; k < ctrstate.maxDepthPerLayer; k++) { // maxDepthPerLayer == 1 means no intralayer edges
                if (layer == 0 && k == 0) continue; // no previous layer available

                #if defined(_OPENMP)
                #pragma omp parallel for
                #endif
                for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi++) {
                    const int threadId = getThreadId();
                    const int v = vi + layer * ctrstate.numNodesPerLayer * ctrstate.maxDepthPerLayer + k * ctrstate.numNodesPerLayer;
                    const int actualV = vi + layer * ctrstate.numNodesPerLayer;
                    for (const auto& it : ctrstate.vertex2preceedingEdgeMap.at(actualV)) {
                        const long e = it;
                        if (e == -1) {
                            continue;
                        }
                        PRECISION_CTR newCost = ctrstate.cost(e) - ctrstate.currentRatio * ctrstate.time(e);
                        long srcIdx = SRCIds(e);
                        const bool isIntraLayerEdge = (srcIdx / ctrstate.numNodesPerLayer) == layer;

                        if (isIntraLayerEdge && k != 0) {
                            if (ctrstate.maxDepthPerLayer <= 1) continue;
                            const int srcIdxInLayer = srcIdx - layer * ctrstate.numNodesPerLayer;
                            if (std::isinf(ctrstate.DPtable(kOffsetCurrentLayer + k - 1, srcIdxInLayer))) continue;

                            const PRECISION_CTR newVal = ctrstate.DPtable(kOffsetCurrentLayer + k - 1, srcIdxInLayer) + newCost;
                            if ( newVal < ctrstate.DPtable(kOffsetCurrentLayer + k, vi) && newVal < ctrstate.upperBound) {
                                anyPathImproves(threadId) = true;
                                const int predecessorIndex = srcIdxInLayer + layer * ctrstate.numNodesPerLayer * ctrstate.maxDepthPerLayer + (k-1) * ctrstate.numNodesPerLayer;
                                ctrstate.predecessor(v) = predecessorIndex;
                                ctrstate.DPtable(kOffsetCurrentLayer + k, vi) = newVal;
                            }
                        }
                        if (!isIntraLayerEdge && k == 0) { // we need to connect to all layer depth extensions
                            for (int kk = 0; kk < ctrstate.maxDepthPerLayer; kk++) {
                                const int srcIdxInLayer = srcIdx - (layer-1) * ctrstate.numNodesPerLayer;
                                if (std::isinf(ctrstate.DPtable(kOffsetPreviousLayer + kk, srcIdxInLayer))) continue;

                                const PRECISION_CTR newVal = ctrstate.DPtable(kOffsetPreviousLayer + kk, srcIdxInLayer) + newCost;
                                if ( newVal < ctrstate.DPtable(kOffsetCurrentLayer, vi) && newVal < ctrstate.upperBound) {
                                    anyPathImproves(threadId) = true;
                                    const int predecessorIndex = srcIdxInLayer + (layer-1) * ctrstate.numNodesPerLayer * ctrstate.maxDepthPerLayer + kk * ctrstate.numNodesPerLayer;
                                    ctrstate.predecessor(v) = predecessorIndex;
                                    ctrstate.DPtable(kOffsetCurrentLayer, vi) = newVal;
                                }
                            }
                        }
                    }
                } // nodes per layer loop (vloop)
            } // k loop


            if (!(anyPathImproves.any()) && layer < ctrstate.numLayers && layer > 0) {
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
            std::vector<int> cycle; cycle.reserve(maxLength);
            const int firstV = indexNodeInLayer + ctrstate.numNodesPerLayer * ctrstate.numLayers * ctrstate.maxDepthPerLayer;
            cycle.push_back(getOriginalGraphIndex(ctrstate, firstV));
            int pred = ctrstate.predecessor(firstV);
            cycle.push_back(getOriginalGraphIndex(ctrstate, pred));
            for (int kk = maxLength - 1; kk > 0; kk--) {
                if (pred == -1) break;
                pred = ctrstate.predecessor(pred);

                cycle.push_back(getOriginalGraphIndex(ctrstate, pred));
                if (pred == indexNodeInLayer) break;
            }
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
        anyPathImproves.setConstant(false);
        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi++) {
            if (ctrstate.nodeBranchId(vi) != ctrstate.branchId) continue;
            const int threadId = getThreadId();
            const int vOnLastLayer = ctrstate.numNodesPerLayer * ctrstate.maxDepthPerLayer * ctrstate.numLayers + vi;
            if (ctrstate.DPtable(kIndexLastLayer, vi) < ctrstate.DPtable(kIndexZeroLayerBackup, vi)) {
                anyPathImproves(threadId) = true;
                ctrstate.DPtable(0, vi) = ctrstate.DPtable(kIndexLastLayer, vi);
                ctrstate.predecessor(vi) = ctrstate.predecessor(vOnLastLayer); // it is not "vOnLastLayer" as we would have zero layer index twice then
            }
            else {
                ctrstate.DPtable(0, vi) = ctrstate.DPtable(kIndexZeroLayerBackup, vi);
            }
        }

        if (!(anyPathImproves.any())) {
            if (ctrstate.verbose)  std::cout << ctrstate.prefix << "      > No path improved => no negative cycle exists" << std::endl;
            const auto outOld = computeRatioForCycle(ctrstate, ctrstate.negativeCycle, productspace, SRCIds, TRGTIds);
            const int integralCycleCostOld = std::round(std::get<1>(outOld));
            if (integralCycleCostOld == 0) {
                if (ctrstate.verbose) std::cout << ctrstate.prefix << "      > Zero cycle found from previous iteration with length: " << ctrstate.negativeCycle.size() << std::endl;
                return BELLMANFORDSTATUS::ZERO_CYCLE;
            }
            return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
        }


        ctrstate.lowerBounds.clear();
        for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi++) {
            if (ctrstate.nodeBranchId(vi) != ctrstate.branchId) continue;
            const PRECISION_CTR lbcandidate = ctrstate.DPtable(0, vi);
            ctrstate.lowerBounds.push_back(std::make_tuple(lbcandidate, vi));
        }
        std::sort(ctrstate.lowerBounds.begin(), ctrstate.lowerBounds.end(), smaller);

        // check for negative cycle
        volatile bool existsNegativeCycle = false;
        int vOnFirstLayerInCycle = -1, cycleLength = -1;
        const int maxCycleLength = ctrstate.numNodes;
        #if defined(_OPENMP)
        #pragma omp parallel for shared(existsNegativeCycle)
        #endif
        //for (int vi = 0; vi < ctrstate.numNodesPerLayer; vi++) {
        for (const LBTUPLE& lb : ctrstate.lowerBounds) {
            const PRECISION_CTR lowerBound = std::get<0>(lb);
            const int vi = std::get<1>(lb);
            if (ctrstate.nodeBranchId(vi) != ctrstate.branchId) continue;
            if (existsNegativeCycle) {
                continue;
            }
            //if (!searchForZeroCycle && ctrstate.DPtable(0, vi) >= -ctrstate.tolerance) {
            if (!searchForZeroCycle && ctrstate.DPtable(0, vi) >= 0) {
                continue;
            }
            if (searchForZeroCycle && (ctrstate.DPtable(0, vi) <= -ctrstate.tolerance || ctrstate.DPtable(0, vi) >= ctrstate.tolerance) ) {
                continue;
            }
            std::vector<int> cycle;
            const int threadId = getThreadId();
            ctrstate.vertexInCycleEncounter.col(threadId).setConstant(false);

            int pred = ctrstate.predecessor(vi);
            ctrstate.vertexInCycleEncounter(vi, threadId) = true;
            int cLength = 1;
            for (int kk = maxCycleLength - 1; kk > 0; kk--) {
                if (pred < ctrstate.numNodesPerLayer && pred >= 0) {
                    if (ctrstate.vertexInCycleEncounter(pred, threadId)) { // we already have seen this vertex
                        #if defined(_OPENMP)
                        #pragma omp critical
                        #endif
                        {
                            existsNegativeCycle = true;
                            vOnFirstLayerInCycle = pred;
                            cycleLength = cLength;
                        }
                        break;
                    }
                    ctrstate.vertexInCycleEncounter(pred, threadId) = true;
                }
                if (existsNegativeCycle) break; // other threads could have found a negative cycle already
                pred = ctrstate.predecessor(pred);
                if (pred == -1) break;
                cLength++;
            }

        }

        // extract negative cycle if one is found
        if (existsNegativeCycle) {
            negativeCycleFound = true;
            std::vector<int> cycle; cycle.reserve(cycleLength);
            int pred = ctrstate.predecessor(vOnFirstLayerInCycle);
            cycle.push_back(getOriginalGraphIndex(ctrstate, pred));
            int k = 0;
            for (int kk = cycleLength - 1; kk > 0; kk--) {
                pred = ctrstate.predecessor(pred);

                cycle.push_back(getOriginalGraphIndex(ctrstate, pred));
                if (pred == vOnFirstLayerInCycle) break;
            }
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
    } // while(!negativeCycleFound)
    return BELLMANFORDSTATUS::NO_NEGATIVE_CYCLE;
}
/*






 LAWLERS ALGORITHM WITH LINEAR SEARCH






 */
bool lawlerSoubroutine(CTRCPUSTATE& ctrstate,
                       const Eigen::MatrixXi& productspace,
                       const Eigen::MatrixXi& SRCIds,
                       const Eigen::MatrixXi& TRGTIds) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    int i = 0;
    const int maxiter = 1000;
    bool existsNegativeCycle = true;
    if (ctrstate.verbose) std::cout << ctrstate.prefix << "     + Running Lawlers alogrithm.. maxiter = " << maxiter << std::endl;
    PRECISION_CTR oldRatio = ctrstate.currentRatio;
    while (existsNegativeCycle) {
        if (ctrstate.verbose) std::cout << ctrstate.prefix << "     + Lawler iter = " << i+1 << ", current ratio = " << ctrstate.currentRatio << std::endl;
        const BELLMANFORDSTATUS bfstatus = mooreBellmanFordSubroutineCPU(ctrstate, productspace, SRCIds, TRGTIds, false);
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

/*






 INITIAL UPPERBOUND COMPUTATION






 */
std::tuple<std::vector<int>, float> CostTimeRatioSolver::findInitalUpperBound(CTRCPUSTATE& ctrstate) {
    if (verbose) std::cout << ctrstate.prefix << "Finding initial upperbound..." << std::endl;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    // find best starting vertex for upperbound computation
    float bestCost = std::numeric_limits<float>::infinity();
    int bestV = 0;
    for (int v = 0; v < ctrstate.numNodesPerLayer; v++) {
        int numIncEdges = 0;
        float mean = 0;
        for (const auto& it : ctrstate.vertex2preceedingEdgeMap.at(v + ctrstate.numNodesPerLayer)) {
            const long e = it;
            if (e == -1) continue;
            mean += ctrstate.cost(e);
            numIncEdges++;
        }
        if (DEBUG_CRT_CYCLE_SOLVER && numIncEdges == 0)
            std::cout << "This node has no incoming edges" << std::endl;
        mean /= numIncEdges;
        if (mean < bestCost && numIncEdges > 3) {
            bestCost = mean;
            bestV = v;
        }
    }

    ctrstate.upperBound = std::numeric_limits<float>::infinity();
    ctrstate.currentRatio = 10;
    ctrstate.nodeBranchId.setOnes();
    ctrstate.nodeBranchId(bestV) = 0;
    ctrstate.branchId = 0;
    const bool verboseBackup = ctrstate.verbose;
    ctrstate.verbose = true;


    BELLMANFORDSTATUS status = mooreBellmanFordSubroutineCPU(ctrstate, productspace, SRCIds, TRGTIds, true, true);

    std::tuple<float, float, Eigen::MatrixXi> ratioAndWeight = computeRatioForCurrentCycle(ctrstate, productspace, SRCIds, TRGTIds);
    PRECISION_CTR cycleCost = std::get<1>(ratioAndWeight);
    while (cycleCost > 0 || std::isnan(cycleCost)) {
        if (verboseBackup) std::cout << ctrstate.prefix << " > did not find suitable upperbound yet, trying ratio "<< ctrstate.currentRatio << std::endl;
        ctrstate.currentRatio += ctrstate.currentRatio;
        ratioAndWeight = computeRatioForCurrentCycle(ctrstate, productspace, SRCIds, TRGTIds);
        cycleCost = std::get<1>(ratioAndWeight);
    }
    ctrstate.currentRatio = std::get<0>(ratioAndWeight);

    ctrstate.nodeBranchId.setZero();
    ctrstate.verbose = verboseBackup;

    //const std::tuple<float, float, Eigen::MatrixXi> ratioAndWeight = computeRatioForCurrentCycle(ctrstate, productspace, SRCIds, TRGTIds);
    //ctrstate.currentRatio = std::get<0>(ratioAndWeight);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << " Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "  [ms])" << std::endl;
    ctrstate.upperBoundCycle = ctrstate.negativeCycle;
    return std::make_tuple(std::vector<int>(), 1.0);
}

void CostTimeRatioSolver::robustFloatToIntConversion(Eigen::MatrixXd& X, const int colIdx, const PRECISION_CTR float2intscaling) {
    const PRECISION_CTR medianFactor = 2; // = 2 means place the median in the middle of float2intscaling
    const PRECISION_CTR max = X.col(colIdx).maxCoeff();
    const PRECISION_CTR median = std::max(utils::median(X.col(colIdx)), 0.001);
    X.col(colIdx) = ((float2intscaling/(medianFactor * median)) * X.col(colIdx)).array().round();
    // clip larger values
    const PRECISION_CTR maxVal = 100 * float2intscaling;
    for (int i = 0; i < X.rows(); i++) {
        if (X(i, colIdx) > maxVal) {
            X(i, colIdx) = maxVal;
        }
    }
}


/*






 BRANCH AND BOUND WRAPPER AROUND LAWLERS ALGORITHM
 (required to find valid cycle => one that does not wrap around multiple times)






 */

CTR_CYCLE_SOLVER_OUTPUT CostTimeRatioSolver::solveWithBnbCPULawler(const PRECISION_CTR float2intScaling,
                                                                   const long numLayers,
                                                                   const long numNodes) {
    const int maxiter = 100;
    if (verbose) std::cout << prefix << "Building graph" << std::endl;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    CTRCPUSTATE ctrstate;
    /*
     INIT
     */
    const long numTotalNodes = numNodesPerLayer * (numLayers+1) * maxDepth;
    ctrstate.predecessor = Eigen::MatrixXi(numTotalNodes, 1);
    ctrstate.nodeBranchId = Eigen::MatrixXi(numNodesPerLayer, 1); ctrstate.nodeBranchId.setZero();
    ctrstate.DPtable = Eigen::MatrixX<PRECISION_CTR>(maxDepth * 2 + 1, numNodesPerLayer);
    ctrstate.branchGraph = branchGraph;
    ctrstate.numTotalNodes = numTotalNodes;
    ctrstate.prefix = prefix;
    ctrstate.maxBranchId = 0;
    ctrstate.numLayers = numLayers;
    ctrstate.numNodes = numNodes;
    ctrstate.numNodesPerLayer = numNodesPerLayer;
    ctrstate.verbose = verbose;
    ctrstate.lowerBounds.reserve(numNodesPerLayer);
    ctrstate.zeroLayerHits.reserve(20);
    ctrstate.maxDepthPerLayer = maxDepth;
    ctrstate.vertexInCycleEncounter = Eigen::MatrixX<bool>(ctrstate.numNodesPerLayer, omp_thread_count()); // column major


    ctrstate.cost = cost.col(0);
    if (cost.cols() > 1) {
        ctrstate.time = cost.col(1); //((float2intScaling/(medianFactor * medianTime)) * cost.col(1)).array().round().cast<PRECISION_CTR>();
    }
    else {
        ctrstate.time = Eigen::MatrixX<PRECISION_CTR>(cost.rows(), 1);
        ctrstate.time.setConstant(float2intScaling / 2);
        if (verbose) std::cout << prefix << "  > no time cost given => running as minimum mean problem" << std::endl;
    }
    ctrstate.tolerance = tolerance;
    /*
     GRAPH CREATION
     */
    ctrstate.vertex2preceedingEdgeMap = buildingInGraph(numNodes + numNodesPerLayer, numNodes);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << " Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "  [ms])" << std::endl;

    /*
     FINDING INITIAL UPPER BOUND
     */
    findInitalUpperBound(ctrstate);

    /*
     BNB loop
     */
    ctrstate.branchId = 0;
    ctrstate.ratioLowerBound = 0; // cost and time are positive
    ctrstate.ratioUpperBound = ctrstate.currentRatio;
    MinHeap ratioLowerBounds(numNodesPerLayer); ratioLowerBounds.push(0, ctrstate.branchId);
    Eigen::MatrixX<bool> branchContainsValidLowerBound(numNodesPerLayer, 1); branchContainsValidLowerBound.setZero();
    ctrstate.nodeBranchId.setZero();
    tsl::robin_map<int, std::vector<int>> validCyclesMap;
    int exploredBranches = 0;
    if (verbose) std::cout << ctrstate.prefix << "Using Branch and Bound OMP to find optimal cost-time-ratio cycle..." << std::endl;

    while(!ratioLowerBounds.isEmpty()) {
        const int numActiveBranches = ratioLowerBounds.getSize();
        const auto heapElement = ratioLowerBounds.pop();
        ctrstate.branchId = heapElement.second;
        ctrstate.ratioLowerBound = heapElement.first;
        if (branchContainsValidLowerBound(ctrstate.branchId)) {
            if (verbose) std::cout << prefix << "  Found optimum in Branch " << ctrstate.branchId << " (current branch is minimum and minimum is valid cycle)" << std::endl;
            ctrstate.ratioUpperBound = heapElement.first;
            ctrstate.negativeCycle = validCyclesMap.at(ctrstate.branchId);
            if (verbose) std::cout << prefix << "  Recovered negative cycle from previous computations" << std::endl;
            break;
        }

        std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        if (verbose) {
            std::cout << prefix  << "  Branches left: " << numActiveBranches << std::endl;
            const float relgap = (ctrstate.ratioUpperBound - ctrstate.ratioLowerBound);
            std::cout << prefix  << "  Branchid " << ctrstate.branchId << ": gap = " << relgap <<" ( lb = " << ctrstate.ratioLowerBound
            << " ub = " << ctrstate.ratioUpperBound << "   time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t - t1).count() * 0.001 << "[s] )"<< std::endl;
        }
        exploredBranches++;
        ctrstate.currentRatio = ctrstate.ratioUpperBound;
        const bool wrapsAroundMultipleTimes = lawlerSoubroutine(ctrstate, productspace, SRCIds, TRGTIds);


        if (ctrstate.ratioUpperBound < ratioLowerBounds.peak() && !wrapsAroundMultipleTimes) {
            // we found an optimum as no other branch contains a lower bound smaller than best found cycle
            if (verbose) std::cout << prefix << "  Found optimum (current ub < lb in other branches)" << std::endl;
            break;
        }
        if (!wrapsAroundMultipleTimes) {
            ratioLowerBounds.push(ctrstate.ratioLowerBoundOfCurrentBranch, ctrstate.branchId);
            branchContainsValidLowerBound(ctrstate.branchId) = true;
            validCyclesMap.insert({ctrstate.branchId, ctrstate.negativeCycle});
            continue;
        }

        if (ctrstate.numNodesInBranch == 1) {
            if (verbose) std::cout << prefix << "  This branch does not the optimum (only one node left)" << std::endl;
            continue;
        }
        if (verbose) std::cout << prefix << "  Did not find optimum yet, branching..." << std::endl;
        branch(ctrstate, productspace, cost, SRCIds, TRGTIds);
        for (const auto& it : ctrstate.zeroLayerHits) {
            const int indexOfZeroLayerHitVertex = it;
            ratioLowerBounds.push(ctrstate.lowerBoundOfCurrentBranch, ctrstate.nodeBranchId(indexOfZeroLayerHitVertex));
        }
    }


    /*
     Writing outputs
     */
    const std::tuple<float, float, Eigen::MatrixXi> ratioWightAndOutputCycle = computeRatioForCurrentCycle(ctrstate, productspace, SRCIds, TRGTIds, true);
    ctrstate.currentRatio = std::get<0>(ratioWightAndOutputCycle);
    const Eigen::MatrixXi outputEdges = std::get<2>(ratioWightAndOutputCycle);
    std::cout << outputEdges << std::endl;
    std::cout << ctrstate.currentRatio << std::endl;
    return std::make_tuple(true, ctrstate.currentRatio, outputEdges);

}

/*
 */
CTR_CYCLE_SOLVER_OUTPUT CostTimeRatioSolver::run(const std::string& solvername) {
    std::cout << std::fixed;
    const long numNodes = SRCIds.maxCoeff() + 1;
    const long numLayers = numNodes / numNodesPerLayer;
    const PRECISION_CTR minCost = cost.minCoeff();
    const PRECISION_CTR maxCost = cost.maxCoeff();
    //const PRECISION_CTR float2intScaling = std::is_same<float, PRECISION_CTR>::value ? 100000.0 : 10000000000000.0;
    const PRECISION_CTR float2intScaling = std::is_same<float, PRECISION_CTR>::value ? 10000.0 : 100000.0;

    std::cout << prefix << "Robustly rescaling cost and time.." << std::endl;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    robustFloatToIntConversion(cost, 0, float2intScaling);
    if (cost.cols() > 1) {
        robustFloatToIntConversion(cost, 1, float2intScaling);
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::cout << prefix << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " [ms])" << std::endl;

    //const PRECISION_CTR float2intScaling = 10000000.0;

    if (solvername.compare("lawlercpu") == 0) {
        return solveWithBnbCPULawler(float2intScaling, numLayers, numNodes);
    }
    #ifdef WITH_CUDA
    if (solvername.compare("lawlergpu") == 0) {
        return solveWithBnbGPULawler(float2intScaling, numLayers, numNodes);
    }
    #endif
    else {
        std::cout << "Solver >>" << solvername << "<< is not implemented" << std::endl;
        exit(-1);
    }
}


void CostTimeRatioSolver::setBranchGraph(const std::vector<tsl::robin_set<long>>& ibranchGraph) {
    branchGraph = ibranchGraph;
}

} // namespace crtsolver
