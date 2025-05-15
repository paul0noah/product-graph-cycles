//
//  mean_cycle_solver.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 07.10.24
//
#ifdef WITH_CUDA
#include "dijkstra_solver.hpp"

#include <chrono>
#include <string.h>
#include <lemon/list_graph.h>
#include <lemon/list_graph.h>
#include <lemon/dijkstra.h>
#include <lemon/path.h>
#include <lemon/howard_mmc.h>
#include <lemon/hartmann_orlin_mmc.h>
#include <lemon/karp_mmc.h>
#include <helper/utils.hpp>
#include <type_traits>
#include <tsl/robin_map.h>
#include "helper/minheap.hpp"
#include "dijkstra_bnb_subroutine.cuh"
#include "dijkstra_solver.hpp"

namespace dijkstra {    
namespace cuda {


void branch(DKSTRAGPUSTATE& dijkstrastate,
            const Eigen::MatrixXi& productspace,
            const Eigen::MatrixXd& cost,
            const Eigen::MatrixXi& SRCIds,
            const Eigen::MatrixXi& TRGTIds,
            const int maxBranchId) {
    const bool branchGraphNeeded = dijkstrastate.branchGraph.size() > 0;
    const auto& branchGraph = branchGraphNeeded ? dijkstrastate.branchGraph : dijkstrastate.vertex2preceedingEdgeMap;

    MinHeap heap(dijkstrastate.numNodesPerLayer);
    for (int i = 0; i < dijkstrastate.numNodesPerLayer; i++) {
        dijkstrastate.branchTable(0, i) = -1;
        if (i == dijkstrastate.branchingSrcNode || i == dijkstrastate.branchingTargetNode) {
            dijkstrastate.branchTable(0, i) = 1;
            continue;
        }
        if (dijkstrastate.nodeBranchId(i) == dijkstrastate.branchId) {
            heap.push(std::numeric_limits<double>::infinity(), i);
            dijkstrastate.branchTable(0, i) = -1; // node unprocessed
        }
    }
    heap.push(0.0, dijkstrastate.branchingSrcNode);
    heap.push(0.0, dijkstrastate.branchingTargetNode);
    dijkstrastate.nodeBranchId(dijkstrastate.branchingTargetNode) = maxBranchId;

    int counter = 0;
    for (const LBTUPLE& lb : dijkstrastate.lowerBounds) { // lower bounds sorted by their cost
        const int v = lb.v;
        if (v == dijkstrastate.branchingSrcNode) continue;
        if (v == dijkstrastate.branchingTargetNode) continue;
        if (dijkstrastate.nodeBranchId(v) != dijkstrastate.branchId) continue;

        // basically put the vertices with worst lower bound to differnt branch
        if (counter > std::max(1, dijkstrastate.numNodesInBranch / 4)) {
            heap.push(0.1, v);
            dijkstrastate.nodeBranchId(v) = maxBranchId;
        }
        counter++;
    }

    int numElementsInNewBranch = 0, numElementsInOldBranch = 0;
    while(!heap.isEmpty()) {
        const std::pair<double, long> current = heap.pop();
        const int currentNodeId = current.second;
        const float currentNodeCost = current.first;
        dijkstrastate.branchTable(0, currentNodeId) = 1.0; // node is processed
        if (dijkstrastate.nodeBranchId(currentNodeId) == maxBranchId) numElementsInNewBranch++;
        if (dijkstrastate.nodeBranchId(currentNodeId) == dijkstrastate.branchId) numElementsInOldBranch++;

        for (const auto& it : branchGraph.at(currentNodeId)) {
            const long e = it;
            if (e == -1) continue; // edge coming from previous layer
            const int srcId = SRCIds(e);
            if (srcId >= dijkstrastate.numNodesPerLayer && DEBUG_DIJKSTRA_SOLVER) {
                std::cout << "srcId >= dijkstrastate.numNodesPerLayer" << std::endl;
                continue;
            }
            if (dijkstrastate.branchTable(0, srcId) < 0) { // srcId node not yet finished processing
                const float oldValue = heap.peakKey(srcId);
                const float newValue = currentNodeCost + 1.0;
                if (newValue < oldValue) {
                    heap.decrease(srcId, newValue);
                    dijkstrastate.nodeBranchId(srcId) = dijkstrastate.nodeBranchId(currentNodeId);
                    if (DEBUG_DIJKSTRA_SOLVER && dijkstrastate.branchingSrcNode == srcId) {
                        std::cout << "Overwriting branching src node branchid. Should not happen" << std::endl;
                    }
                }
            }
        }
    }

    if (DEBUG_DIJKSTRA_SOLVER) {
        int numinfirstbranch = 0, numinsecondbranch = 0;
        for (int i = 0; i < dijkstrastate.numNodesPerLayer; i++) {
            if (dijkstrastate.nodeBranchId(i) == dijkstrastate.branchId) numinfirstbranch++;
            if (dijkstrastate.nodeBranchId(i) == maxBranchId) numinsecondbranch++;
        }
        if (numinfirstbranch == 0) {
            std::cout << "No vertices in first branch. this should not happen" << std::endl;
        }
        if (numinsecondbranch == 0) {
            std::cout << "No vertices in second branch. this should not happen" << std::endl;
        }
        std::cout << "numinfirstbranch " << numinfirstbranch << "  numinsecondbranch " << numinsecondbranch <<  "    " << dijkstrastate.numNodesInBranch - numinfirstbranch - numinsecondbranch << std::endl;
    }
    dijkstrastate.branchingSrcNode = -1;
    dijkstrastate.branchingTargetNode = -1;
}
} // namespace cuda

/*






 BRANCH AND BOUND WRAPPER AROUND BELLMAN FORD
 (required to find valid cycle => one that does not wrap around multiple times)






 */
DIJKSTRA_SOLVER_OUTPUT DijkstraSolver::solveWithBnBCuda(const long numLayers, const long numNodes) {
    using namespace cuda;
    const int maxiter = 100;
    if (verbose) std::cout << prefix << "Building graph" << std::endl;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    DKSTRAGPUSTATE dijkstrastate;
    /*
     INIT
     */
    const long numTotalNodes = numNodesPerLayer * (numLayers+1) * maxDepth;
    dijkstrastate.branchTable = Eigen::MatrixX<PRECISION_DIJK>(1, numNodesPerLayer);
    dijkstrastate.predecessor = CudaMatrix<int>(numTotalNodes, 1);
    dijkstrastate.nodeBranchId = Eigen::MatrixXi(numNodesPerLayer, 1); dijkstrastate.nodeBranchId.setZero();
    dijkstrastate.nodeBranchIdCuda = CudaMatrix<int>(numNodesPerLayer, 1);
    dijkstrastate.branchTable = Eigen::MatrixX<PRECISION_DIJK>(1, numNodesPerLayer);
    dijkstrastate.DPtable = CudaMatrix<PRECISION_DIJK>(maxDepth * 2 + 1, numNodesPerLayer);
    dijkstrastate.SRCIds = CudaMatrix<int>(SRCIds);
    dijkstrastate.cycleCuda = CYCLE(10 * numLayers * maxDepth);
    dijkstrastate.cost = cost.col(0);
    dijkstrastate.costCuda = CudaMatrix<PRECISION_DIJK>(dijkstrastate.cost);
    dijkstrastate.anyPathImproves = CudaMatrix<bool>(1, 1);
    dijkstrastate.lbarray =  LBARRAY(numNodesPerLayer);

    dijkstrastate.branchGraph = branchGraph;
    dijkstrastate.numTotalNodes = numTotalNodes;
    dijkstrastate.prefix = prefix;
    dijkstrastate.maxBranchId = 0;
    dijkstrastate.numLayers = numLayers;
    dijkstrastate.numNodes = numNodes;
    dijkstrastate.numNodesPerLayer = numNodesPerLayer;
    dijkstrastate.verbose = verbose;
    dijkstrastate.maxDepthPerLayer = maxDepth;
    //dijkstrastate.vertexInCycleEncounter = Eigen::MatrixX<bool>(dijkstrastate.numNodesPerLayer, omp_thread_count()); // column major
    dijkstrastate.upperBound = CUSTOM_CUDA_INF;
    dijkstrastate.lowerBound = 0;
    dijkstrastate.branchId = 0;
    // cuda setup
    const cudaDeviceProp props = initialize_gpu(prefix, verbose);
    const int threadsPerBlock = props.maxThreadsPerMultiProcessor > props.maxThreadsPerBlock ? props.maxThreadsPerMultiProcessor / 2 : props.maxThreadsPerBlock;
    int numBlocks =  props.maxThreadsPerMultiProcessor * props.multiProcessorCount / threadsPerBlock;
    if (numBlocks * threadsPerBlock > dijkstrastate.numNodesPerLayer) { // num nodes per layer is maximum we can do in parallel
        numBlocks = std::ceil(dijkstrastate.numNodesPerLayer / (float) threadsPerBlock);
    }
    if (verbose) std::cout << prefix << "  > setting #blocks = " << numBlocks << " and #threadsPerBlock = " << threadsPerBlock << std::endl;
    dijkstrastate.numThreadBlocks = numBlocks;
    dijkstrastate.threadsPerBlock = threadsPerBlock;
    /*
     GRAPH CREATION
     */
    dijkstrastate.vertex2preceedingEdgeMap = buildingInGraph(numNodes + numNodesPerLayer, numNodes);
    dijkstrastate.vertex2preceedingEdgeMapCuda = VECSET(dijkstrastate.vertex2preceedingEdgeMap);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << " Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "  [ms])" << std::endl;

    /*
     FINDING INITIAL UPPER BOUND
     */


    /*
     MAIN BNB LOOP
     */
    dijkstrastate.nodeBranchId.setZero();
    dijkstrastate.nodeBranchIdCuda.copyEigen(dijkstrastate.nodeBranchId);
    MinHeap lowerBounds(numNodesPerLayer); lowerBounds.push(0.0, dijkstrastate.branchId);
    int maxBranchId = 0;
    Eigen::MatrixX<bool> branchContainsValidLowerBound(numNodesPerLayer, 1); branchContainsValidLowerBound.setZero();
    tsl::robin_map<int, thrust::host_vector<int>> validCyclesMap;
    int exploredBranches = 0;
    while(!lowerBounds.isEmpty()) {
        const int numActiveBranches = lowerBounds.getSize();
        const auto heapElement = lowerBounds.pop();
        dijkstrastate.branchId = heapElement.second;
        dijkstrastate.lowerBound = heapElement.first;
        if (branchContainsValidLowerBound(dijkstrastate.branchId)) {
            if (verbose) std::cout << prefix << "  Found optimum in Branch " << dijkstrastate.branchId << " (current branch is minimum and minimum is valid cycle)" << std::endl;
            dijkstrastate.upperBound = heapElement.first;
            dijkstrastate.cycle = validCyclesMap.at(dijkstrastate.branchId);
            if (verbose) std::cout << prefix << "  Recovered minimum mean cycle from previous computations" << std::endl;
            break;
        }


        std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
        if (verbose) {
            std::cout << prefix  << "  Branches left: " << numActiveBranches << std::endl;
            const float relgap = (dijkstrastate.upperBound - dijkstrastate.lowerBound);
            std::cout << prefix  << "  Branchid " << dijkstrastate.branchId << ": gap = " << relgap <<" ( lb = " << dijkstrastate.lowerBound
            << " ub = " << dijkstrastate.upperBound << " time = " << std::chrono::duration_cast<std::chrono::milliseconds>(t - t1).count() * 0.001 << "[s] )"<< std::endl;
        }

        exploredBranches++;
        const bool branchCouldContainMin = mooreBellmanFordSubroutineGPU(dijkstrastate, productspace, SRCIds, TRGTIds);
        if (!branchCouldContainMin) continue;
        if (dijkstrastate.upperBound < lowerBounds.peak() && dijkstrastate.upperboundFoundOnFirstTry) {
            // we found an optimum as no other branch contains a lower bound smaller than best found cycle
            if (verbose) std::cout << prefix << "  Found optimum (current ub < lb in other branches)" << std::endl;
            break;
        }
        if (dijkstrastate.upperboundFoundOnFirstTry) {
            lowerBounds.push(dijkstrastate.lowerBoundOfCurrentBranch, dijkstrastate.branchId);
            branchContainsValidLowerBound(dijkstrastate.branchId) = true;
            validCyclesMap.insert({dijkstrastate.branchId, dijkstrastate.cycle});
            continue;
        }

        if (verbose) std::cout << prefix << "  Did not find optimum yet, branching..." << std::endl;
        maxBranchId++;
        branch(dijkstrastate, productspace, cost, SRCIds, TRGTIds, maxBranchId);
        lowerBounds.push(dijkstrastate.lowerBoundOfCurrentBranch, dijkstrastate.branchId);
        lowerBounds.push(dijkstrastate.lowerBoundOfCurrentBranch, maxBranchId);
        dijkstrastate.nodeBranchIdCuda.copyEigen(dijkstrastate.nodeBranchId);
    }


    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << prefix << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;
        std::cout << prefix << "Explored " << exploredBranches << " branches to find optimum" << std::endl;
    }


    /*
     PATH OUTPUT
     */
    const int cycleLength = dijkstrastate.cycle.size();

    Eigen::MatrixXi outputEdges(cycleLength+1, productspace.cols()); outputEdges = - outputEdges.setOnes();
    int cycleElementsAdded = 0, numLoopsThroughAllLayers = 0;
    std::reverse(dijkstrastate.cycle.begin(), dijkstrastate.cycle.end());
    dijkstrastate.cycle.push_back(dijkstrastate.cycle.front() + dijkstrastate.numNodes);
    const int loopmax = cycleLength+1;
    for (int i = 0; i < loopmax; i++) {
        const int cycleSrcIdx = (i) % loopmax;
        const int cycleTrgtIdx = (i+1) % loopmax;
        const int srcIdx = dijkstrastate.cycle[cycleSrcIdx];
        const int trgtIdx = dijkstrastate.cycle[cycleTrgtIdx];
        if (DEBUG_DIJKSTRA_SOLVER) std::cout << srcIdx << "-->" << trgtIdx;
        if (trgtIdx >= numNodes) {
            numLoopsThroughAllLayers++;
            //continue;
        }
        long edgeIdx = -2;
        for (const auto& it : dijkstrastate.vertex2preceedingEdgeMap.at(trgtIdx)) {
            const long e = it;
            if (e != -1 && SRCIds(e) == srcIdx) {
                edgeIdx = e;
                break;
            }
            if (e == -1 && trgtIdx == srcIdx - numNodes) {
                edgeIdx = e;
                break;
            }
        }
        if (edgeIdx == -1) {
            if (DEBUG_DIJKSTRA_SOLVER) std::cout << std::endl;
            continue;
        }
        if (edgeIdx == -2) {
            std::cout << prefix << "edgeIdx is -2: did not find the respective edge with srcIdx=" << srcIdx << " trgtIdx=" << trgtIdx << std::endl;
            continue;
        }
        if (DEBUG_DIJKSTRA_SOLVER) std::cout << " (weight: "  << cost(edgeIdx) << ")"  << std::endl;

        outputEdges.row(cycleElementsAdded) = productspace.row(edgeIdx);
        cycleElementsAdded++;
    }

    const bool cyclePassesLayersJustOnce = numLoopsThroughAllLayers < 2;
    outputEdges.conservativeResize(cycleElementsAdded, productspace.cols());
    std::cout << outputEdges << std::endl;



    // cleanup
    dijkstrastate.SRCIds.destroy();
    dijkstrastate.costCuda.destroy();
    dijkstrastate.predecessor.destroy();
    dijkstrastate.DPtable.destroy();
    dijkstrastate.nodeBranchIdCuda.destroy();
    dijkstrastate.anyPathImproves.destroy();
    dijkstrastate.lbarray.destroy();
    dijkstrastate.cycleCuda.destroy();
    dijkstrastate.vertex2preceedingEdgeMapCuda.destroy();
    return std::make_tuple(cyclePassesLayersJustOnce, dijkstrastate.upperBound, outputEdges);

}


} // namespace dijkstra

#endif
