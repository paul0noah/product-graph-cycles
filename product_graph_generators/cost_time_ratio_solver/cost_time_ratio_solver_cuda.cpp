//
//  cost_time_ratio_solver_cuda.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 08.11.24
//
#ifdef WITH_CUDA
#include "cost_time_ratio_solver.hpp"

#include <chrono>
#include <string.h>
#include <helper/utils.hpp>
#include <type_traits>
#include <tsl/robin_map.h>
#include "helper/minheap.hpp"
#include "ctr_bnb_subroutine.cuh"

namespace ctrsolver {
namespace cuda {

void branch(CTRGPUSTATE& ctrstate,
            const Eigen::MatrixXi& productspace,
            const Eigen::MatrixXd& cost,
            const Eigen::MatrixXi& SRCIds,
            const Eigen::MatrixXi& TRGTIds) {
    const bool branchGraphNeeded = ctrstate.branchGraph.size() > 0;
    const auto& branchGraph = branchGraphNeeded ? ctrstate.branchGraph : ctrstate.vertex2preceedingEdgeMap;

    MinHeap heap(ctrstate.numNodesPerLayer);
    for (int i = 0; i < ctrstate.numNodesPerLayer; i++) {
        ctrstate.branchTable(0, i) = -1;
        if (i == ctrstate.branchingSrcNode || i == ctrstate.branchingTargetNode) {
            ctrstate.branchTable(0, i) = 1;
            continue;
        }
        if (ctrstate.branchTable(i) == ctrstate.branchId) {
            heap.push(std::numeric_limits<double>::infinity(), i);
            ctrstate.branchTable(0, i) = -1; // node unprocessed
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

    while(!heap.isEmpty()) {
        const std::pair<double, long> current = heap.pop();
        const int currentNodeId = current.second;
        const float currentNodeCost = current.first;
        ctrstate.branchTable(0, currentNodeId) = 1.0; // node is processed

        for (const auto& it : branchGraph.at(currentNodeId)) {
            const long e = it;
            if (e == -1) continue; // edge coming from previous layer
            const int srcId = SRCIds(e);
            if (ctrstate.branchTable(0, srcId) < 0) { // srcId node not yet finished processing
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






 INITIAL UPPERBOUND COMPUTATION






 */
std::tuple<std::vector<int>, float> findInitalUpperBound(CTRGPUSTATE& ctrstate,
                                                        const Eigen::MatrixXi& productspace,
                                                        const Eigen::MatrixXi& SRCIds,
                                                        const Eigen::MatrixXi& TRGTIds) {
    if (ctrstate.verbose) std::cout << ctrstate.prefix << "Finding initial upperbound..." << std::endl;
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
    ctrstate.nodeBranchIdCuda.copyEigen(ctrstate.nodeBranchId);
    ctrstate.branchId = 0;
    const bool verboseBackup = ctrstate.verbose;
    ctrstate.verbose = true;

    BELLMANFORDSTATUS status = mooreBellmanFordSubroutineGPU(ctrstate, productspace, SRCIds, TRGTIds, true, true);

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
    ctrstate.nodeBranchIdCuda.copyEigen(ctrstate.nodeBranchId);
    ctrstate.verbose = verboseBackup;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (ctrstate.verbose) std::cout << ctrstate.prefix << " Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "  [ms])" << std::endl;
    ctrstate.upperBoundCycle = ctrstate.negativeCycle;
    return std::make_tuple(std::vector<int>(), 1.0);
}
} // namespace cuda

/*






 BRANCH AND BOUND WRAPPER AROUND LAWLERS ALGORITHM
 (required to find valid cycle => one that does not wrap around multiple times)






 */

CTR_CYCLE_SOLVER_OUTPUT CostTimeRatioSolver::solveWithBnbGPULawler(const PRECISION_CTR float2intScaling,
                                                                   const long numLayers,
                                                                   const long numNodes) {
    using namespace cuda;
    const int maxiter = 100;
    if (verbose) std::cout << prefix << "Building graph" << std::endl;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    CTRGPUSTATE ctrstate;
    /*
     INIT
     */
    const long numTotalNodes = numNodesPerLayer * (numLayers+1) * maxDepth;
    // TODO set colmajority here
    ctrstate.predecessor = CudaMatrix<int>(numTotalNodes, 1);
    ctrstate.nodeBranchId = Eigen::MatrixXi(numNodesPerLayer, 1); ctrstate.nodeBranchId.setZero();
    ctrstate.nodeBranchIdCuda = CudaMatrix<int>(numNodesPerLayer, 1);
    ctrstate.branchTable = Eigen::MatrixX<PRECISION_CTR>(1, numNodesPerLayer);
    ctrstate.DPtable = CudaMatrix<PRECISION_CTR>(maxDepth * 2 + 1, numNodesPerLayer);
    ctrstate.SRCIds = CudaMatrix<int>(SRCIds);
    ctrstate.cycleCuda = CYCLE(100 * numLayers * maxDepth); // this could lead to crashes if cycle is longer
    ctrstate.branchGraph = branchGraph;
    ctrstate.numTotalNodes = numTotalNodes;
    ctrstate.prefix = prefix;
    ctrstate.maxBranchId = 0;
    ctrstate.numLayers = numLayers;
    ctrstate.numNodes = numNodes;
    ctrstate.numNodesPerLayer = numNodesPerLayer;
    ctrstate.verbose = verbose;
    ctrstate.zeroLayerHits.reserve(20);
    ctrstate.maxDepthPerLayer = maxDepth;
    ctrstate.vertexInCycleEncounter = CudaMatrix<bool>(THREAD_VERTEX_CYCLE_ENCOUNTER, ctrstate.numNodesPerLayer); // row major
    ctrstate.cycleDetectionHelper = CudaMatrix<int>(THREAD_VERTEX_CYCLE_ENCOUNTER, 3);
    ctrstate.cost = cost.col(0);
    ctrstate.costCuda = CudaMatrix<PRECISION_CTR>(ctrstate.cost);
    if (cost.cols() > 1) {
        ctrstate.time = cost.col(1);
    }
    else {
        ctrstate.time = Eigen::MatrixX<PRECISION_CTR>(cost.rows(), 1); 
        ctrstate.time.setConstant(float2intScaling / 2);
        if (verbose) std::cout << prefix << "  > no time cost given => running as minimum mean problem" << std::endl;
    }
    ctrstate.timeCuda = CudaMatrix<PRECISION_CTR>(ctrstate.time);
    ctrstate.tolerance = tolerance;
    // cuda setup
    const cudaDeviceProp props = initialize_gpu(prefix, verbose);
    const int threadsPerBlock = props.maxThreadsPerMultiProcessor > props.maxThreadsPerBlock ? props.maxThreadsPerMultiProcessor / 2 : props.maxThreadsPerBlock;
    int numBlocks =  props.maxThreadsPerMultiProcessor * props.multiProcessorCount / threadsPerBlock;
    if (numBlocks * threadsPerBlock > ctrstate.numNodesPerLayer) { // num nodes per layer is maximum we can do in parallel
        numBlocks = std::ceil(ctrstate.numNodesPerLayer / (float) threadsPerBlock);
    }
    if (verbose) std::cout << prefix << "  > setting #blocks = " << numBlocks << " and #threadsPerBlock = " << threadsPerBlock << std::endl;
    ctrstate.numThreadBlocks = numBlocks;
    ctrstate.threadsPerBlock = threadsPerBlock;
    ctrstate.anyPathImproves = CudaMatrix<bool>(1, 1);

    /*
     GRAPH CREATION
     */
    ctrstate.vertex2preceedingEdgeMap = buildingInGraph(numNodes + numNodesPerLayer, numNodes);
    ctrstate.vertex2preceedingEdgeMapCuda = VECSET(ctrstate.vertex2preceedingEdgeMap);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << " Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "  [ms])" << std::endl;

    /*
     FINDING INITIAL UPPER BOUND
     */
    cuda::findInitalUpperBound(ctrstate, productspace, SRCIds, TRGTIds);

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
    if (verbose) std::cout << ctrstate.prefix << "Using Branch and Bound CUDA to find optimal cost-time-ratio cycle..." << std::endl;

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
        const bool wrapsAroundMultipleTimes = cuda::lawlerSoubroutine(ctrstate, productspace, SRCIds, TRGTIds);


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
        ctrstate.nodeBranchIdCuda.copyEigen(ctrstate.nodeBranchId);
    }


    /*
     Writing outputs
     */
    const std::tuple<float, float, Eigen::MatrixXi> ratioWightAndOutputCycle = computeRatioForCurrentCycle(ctrstate, productspace, SRCIds, TRGTIds, true);
    ctrstate.currentRatio = std::get<0>(ratioWightAndOutputCycle);
    const Eigen::MatrixXi outputEdges = std::get<2>(ratioWightAndOutputCycle);
    std::cout << outputEdges << std::endl;
    std::cout << ctrstate.currentRatio << std::endl;
    
    // cleaning up
    ctrstate.SRCIds.destroy();;
    ctrstate.costCuda.destroy();;
    ctrstate.timeCuda.destroy();;
    ctrstate.predecessor.destroy();;
    ctrstate.DPtable.destroy();;
    ctrstate.vertexInCycleEncounter.destroy();;
    ctrstate.cycleDetectionHelper.destroy();;
    ctrstate.nodeBranchIdCuda.destroy();;
    ctrstate.anyPathImproves.destroy();;
    ctrstate.cycleCuda.destroy();
    ctrstate.vertex2preceedingEdgeMapCuda.destroy();
    return std::make_tuple(true, ctrstate.currentRatio, outputEdges);

}
} // namespace crtsolver

#endif
