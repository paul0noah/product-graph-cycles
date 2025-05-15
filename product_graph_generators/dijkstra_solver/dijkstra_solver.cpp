//
//  mean_cycle_solver.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 07.10.24
//
#include "dijkstra_solver.hpp"

#include <chrono>
#include <string.h>
#include <helper/utils.hpp>
#include <type_traits>
#include <tsl/robin_map.h>
#include "helper/minheap.hpp"

namespace dijkstra {
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

std::vector<tsl::robin_set<long>> DijkstraSolver::buildingInGraph(const long numTotalNodes,
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

DijkstraSolver::DijkstraSolver(const Eigen::MatrixXi& productspace,
                               Eigen::MatrixXd& cost,
                               const Eigen::MatrixXi& SRCIds,
                               const Eigen::MatrixXi& TRGTIds,
                               const long numNodesPerLayer,
                               const int imaxDepth) : productspace(productspace), cost(cost), SRCIds(SRCIds), TRGTIds(TRGTIds), numNodesPerLayer(numNodesPerLayer) {
    verbose = true;
    prefix = "[DSTRSolver] ";
    branchGraph = std::vector<tsl::robin_set<long>>();
    cycleChecking = true;
    maxDepth = imaxDepth;
    if(verbose) std::cout << prefix << "Running DijkstraSolver with maxDepthPerLayer = " << maxDepth << std::endl;

}

DijkstraSolver::~DijkstraSolver() {

}


void branch(DKSTRACPUSTATE& dijkstrastate,
            const Eigen::MatrixXi& productspace,
            const Eigen::MatrixXd& cost,
            const Eigen::MatrixXi& SRCIds,
            const Eigen::MatrixXi& TRGTIds,
            const int maxBranchId) {
    const bool branchGraphNeeded = dijkstrastate.branchGraph.size() > 0;
    const auto& branchGraph = branchGraphNeeded ? dijkstrastate.branchGraph : dijkstrastate.vertex2preceedingEdgeMap;

    MinHeap heap(dijkstrastate.numNodesPerLayer);
    for (int i = 0; i < dijkstrastate.numNodesPerLayer; i++) {
        dijkstrastate.DPtable(0, i) = -1;
        if (i == dijkstrastate.branchingSrcNode || i == dijkstrastate.branchingTargetNode) {
            dijkstrastate.DPtable(0, i) = 1;
            continue;
        }
        if (dijkstrastate.nodeBranchId(i) == dijkstrastate.branchId) {
            heap.push(std::numeric_limits<double>::infinity(), i);
            dijkstrastate.DPtable(0, i) = -1; // node unprocessed
        }
    }
    heap.push(0.0, dijkstrastate.branchingSrcNode);
    heap.push(0.0, dijkstrastate.branchingTargetNode);
    dijkstrastate.nodeBranchId(dijkstrastate.branchingTargetNode) = maxBranchId;

    int counter = 0;
    for (const LBTUPLE& lb : dijkstrastate.lowerBounds) { // lower bounds sorted by their cost
        const int v = std::get<1>(lb);
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
        dijkstrastate.DPtable(0, currentNodeId) = 1.0; // node is processed
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
            if (dijkstrastate.DPtable(0, srcId) < 0) { // srcId node not yet finished processing
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


/*






 Moore-Bellman-Ford subroutine






 */
int getOriginalGraphIndex(const DKSTRACPUSTATE& dijkstrastate, const int pred) {
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
bool mooreBellmanFordSubroutineCPU(DKSTRACPUSTATE& dijkstrastate,
                                const Eigen::MatrixXi& productspace,
                                const Eigen::MatrixXi& SRCIds,
                                const Eigen::MatrixXi& TRGTIds,
                                const bool searchForUpperBound=false) {
    #if defined(_OPENMP)
    const int numThreads = omp_thread_count();
    #else
    const int numThreads = 1;
    #endif
    // setup
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    Eigen::MatrixXf perVertexLowerBounds(dijkstrastate.numNodesPerLayer, numThreads);
    perVertexLowerBounds.setConstant(std::numeric_limits<float>::infinity());
    Eigen::MatrixXi perVertexLbK(dijkstrastate.numNodesPerLayer, numThreads);
    dijkstrastate.DPtable.setConstant(std::numeric_limits<float>::infinity());

    int indexNodeInLayer = -1;
    int numNodesInBranch = 0;
    #if defined(_OPENMP)
    #pragma omp parallel for reduction (+:numNodesInBranch)
    #endif
    for (int i = 0; i < dijkstrastate.nodeBranchId.rows(); i++) {
        const int startVertexId = i;
        if (dijkstrastate.nodeBranchId(startVertexId) == dijkstrastate.branchId) {
            indexNodeInLayer = startVertexId;
            numNodesInBranch++;
            dijkstrastate.DPtable(0, startVertexId) = 0;
            dijkstrastate.predecessor(startVertexId) = -1;
        }
    }
    if (dijkstrastate.verbose) std::cout << dijkstrastate.prefix << "      > Num nodes in branch " << numNodesInBranch << std::endl;
    dijkstrastate.numNodesInBranch = numNodesInBranch;
    if (numNodesInBranch == 0) {
        std::cout << "NO NODES IN BRANCH, this should not happen" << std::endl; exit(-1);
    }

    // dynamic programming loop: sweep through all layers
    Eigen::MatrixX<bool> anyPathImproves(numThreads, 1); anyPathImproves.setConstant(true);
    bool negativeCycleFound = false;
    int iter = 0;
    const int kIndexZeroLayerBackup  = 2 * dijkstrastate.maxDepthPerLayer;
    dijkstrastate.DPtable.row(kIndexZeroLayerBackup) = dijkstrastate.DPtable.row(0); // backup zerolayer values


    for (int layer = 0; layer < dijkstrastate.numLayers + 1; layer++) {
        const int kOffsetPreviousLayer = ((layer+1) % 2) * dijkstrastate.maxDepthPerLayer;
        const int kOffsetCurrentLayer = (layer % 2) * dijkstrastate.maxDepthPerLayer;
        if (layer != 0) {
            dijkstrastate.DPtable.block(kOffsetCurrentLayer, 0, dijkstrastate.maxDepthPerLayer, dijkstrastate.numNodesPerLayer).setConstant(std::numeric_limits<float>::infinity());
            anyPathImproves.setConstant(false);
        }

        for (int k = 0; k < dijkstrastate.maxDepthPerLayer; k++) { // maxDepthPerLayer == 1 means no intralayer edges
            if (layer == 0 && k == 0) continue; // no previous layer available

            #if defined(_OPENMP)
            #pragma omp parallel for
            #endif
            for (int vi = 0; vi < dijkstrastate.numNodesPerLayer; vi++) {
                const int threadId = getThreadId();
                const int v = vi + layer * dijkstrastate.numNodesPerLayer * dijkstrastate.maxDepthPerLayer + k * dijkstrastate.numNodesPerLayer;
                const int actualV = vi + layer * dijkstrastate.numNodesPerLayer;
                for (const auto& it : dijkstrastate.vertex2preceedingEdgeMap.at(actualV)) {
                    const long e = it;
                    if (e == -1) {
                        continue;
                    }
                    PRECISION_DIJK newCost = dijkstrastate.cost(e);
                    long srcIdx = SRCIds(e);
                    const bool isIntraLayerEdge = (srcIdx / dijkstrastate.numNodesPerLayer) == layer;

                    if (isIntraLayerEdge && k != 0) {
                        if (dijkstrastate.maxDepthPerLayer <= 1) continue;
                        const int srcIdxInLayer = srcIdx - layer * dijkstrastate.numNodesPerLayer;
                        if (std::isinf(dijkstrastate.DPtable(kOffsetCurrentLayer + k - 1, srcIdxInLayer))) continue;

                        const PRECISION_DIJK newVal = dijkstrastate.DPtable(kOffsetCurrentLayer + k - 1, srcIdxInLayer) + newCost;
                        if ( newVal < dijkstrastate.DPtable(kOffsetCurrentLayer + k, vi) && newVal < dijkstrastate.upperBound) {
                            anyPathImproves(threadId) = true;
                            const int predecessorIndex = srcIdxInLayer + layer * dijkstrastate.numNodesPerLayer * dijkstrastate.maxDepthPerLayer + (k-1) * dijkstrastate.numNodesPerLayer;
                            dijkstrastate.predecessor(v) = predecessorIndex;
                            dijkstrastate.DPtable(kOffsetCurrentLayer + k, vi) = newVal;
                        }
                    }
                    if (!isIntraLayerEdge && k == 0) { // we need to connect to all layer depth extensions
                        for (int kk = 0; kk < dijkstrastate.maxDepthPerLayer; kk++) {
                            const int srcIdxInLayer = srcIdx - (layer-1) * dijkstrastate.numNodesPerLayer;
                            if (std::isinf(dijkstrastate.DPtable(kOffsetPreviousLayer + kk, srcIdxInLayer))) continue;

                            const PRECISION_DIJK newVal = dijkstrastate.DPtable(kOffsetPreviousLayer + kk, srcIdxInLayer) + newCost;
                            if ( newVal < dijkstrastate.DPtable(kOffsetCurrentLayer, vi) && newVal < dijkstrastate.upperBound) {
                                anyPathImproves(threadId) = true;
                                const int predecessorIndex = srcIdxInLayer + (layer-1) * dijkstrastate.numNodesPerLayer * dijkstrastate.maxDepthPerLayer + kk * dijkstrastate.numNodesPerLayer;
                                dijkstrastate.predecessor(v) = predecessorIndex;
                                dijkstrastate.DPtable(kOffsetCurrentLayer, vi) = newVal;
                            }
                        }
                    }
                }
            } // nodes per layer loop (vloop)
        } // k loop


        if (!(anyPathImproves.any()) && layer < dijkstrastate.numLayers && layer > 0) {
            if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "      > No path improved, early stopping" << std::endl;
            return false;
        }
    } // layer loop

    // check if layer zero has improvement and rewrite zero layer in DP table

    /*
     Output part
     */
    const int kIndexLastLayer  = dijkstrastate.numLayers % 2; // this contains so far optimal values of duplicated zeroth layer

    dijkstrastate.lowerBounds.clear();
    bool anyPotentialNewLowerBound = false;
    for (int vi = 0; vi < dijkstrastate.numNodesPerLayer; vi++) {
        if (dijkstrastate.nodeBranchId(vi) != dijkstrastate.branchId) continue;
        const PRECISION_DIJK lbcandidate = dijkstrastate.DPtable(kIndexLastLayer, vi);
        dijkstrastate.lowerBounds.push_back(std::make_tuple(lbcandidate, vi));
        if (!isinf(lbcandidate)) {
            anyPotentialNewLowerBound = true;
        }
    }
    if (!anyPotentialNewLowerBound) {
        if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "      > No non-inf lower bound" << std::endl;
        return false;
    }
    std::sort(dijkstrastate.lowerBounds.begin(), dijkstrastate.lowerBounds.end(), smaller);


    std::sort(dijkstrastate.lowerBounds.begin(), dijkstrastate.lowerBounds.end(), smaller);
    bool firstIterPathSearchLoop = true;
    const int maxCycleLength = dijkstrastate.numLayers * dijkstrastate.maxDepthPerLayer;
    std::vector<int> cycle; cycle.reserve(maxCycleLength);
    dijkstrastate.upperboundFoundOnFirstTry = false;
    for (const LBTUPLE& lb : dijkstrastate.lowerBounds) {
        const float lowerBound = std::get<0>(lb);
        if (std::isinf(lowerBound)) {
            firstIterPathSearchLoop = false;
            continue;
        }
        const int v = std::get<1>(lb);
        const int vOnLastLayer = v + dijkstrastate.numNodesPerLayer * dijkstrastate.numLayers * dijkstrastate.maxDepthPerLayer;
        cycle.clear();
        int pred = dijkstrastate.predecessor(vOnLastLayer);
        cycle.push_back(getOriginalGraphIndex(dijkstrastate, pred));
        for (int kk = maxCycleLength; kk > 0; kk--) {
            pred = dijkstrastate.predecessor(pred);
            if (pred == -1) break;
            cycle.push_back(getOriginalGraphIndex(dijkstrastate, pred));
        }

        if (firstIterPathSearchLoop) {
            dijkstrastate.lowerBoundOfCurrentBranch = lowerBound;
        }
        if (v == cycle.back()) { // the path starts and ends in the same vertex => lower bound is a valid path
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
                dijkstrastate.cycle = cycle;
            }
            break;
        }
        else if (firstIterPathSearchLoop) {
            if (dijkstrastate.verbose)  std::cout << dijkstrastate.prefix << "    > best lower bound in branch " << lowerBound << std::endl;
            dijkstrastate.branchingSrcNode = cycle.front() - dijkstrastate.numNodes;
            dijkstrastate.branchingTargetNode = cycle.back();
        }
        firstIterPathSearchLoop = false; // only first iteration ;)
    }

    return true;
}


/*






 BRANCH AND BOUND WRAPPER AROUND BELLMAN FORD
 (required to find valid cycle => one that does not wrap around multiple times)






 */
DIJKSTRA_SOLVER_OUTPUT DijkstraSolver::solveWithBnB(const long numLayers, const long numNodes) {
    const int maxiter = 100;
    if (verbose) std::cout << prefix << "Building graph" << std::endl;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    DKSTRACPUSTATE dijkstrastate;
    /*
     INIT
     */
    const long numTotalNodes = numNodesPerLayer * (numLayers+1) * maxDepth;
    dijkstrastate.predecessor = Eigen::MatrixXi(numTotalNodes, 1);
    dijkstrastate.nodeBranchId = Eigen::MatrixXi(numNodesPerLayer, 1);
    dijkstrastate.DPtable = Eigen::MatrixX<PRECISION_DIJK>(maxDepth * 2 + 1, numNodesPerLayer);
    dijkstrastate.branchGraph = branchGraph;
    dijkstrastate.numTotalNodes = numTotalNodes;
    dijkstrastate.prefix = prefix;
    dijkstrastate.maxBranchId = 0;
    dijkstrastate.numLayers = numLayers;
    dijkstrastate.numNodes = numNodes;
    dijkstrastate.numNodesPerLayer = numNodesPerLayer;
    dijkstrastate.verbose = verbose;
    dijkstrastate.lowerBounds.reserve(numNodesPerLayer);
    dijkstrastate.zeroLayerHits.reserve(20);
    dijkstrastate.maxDepthPerLayer = maxDepth;
    dijkstrastate.vertexInCycleEncounter = Eigen::MatrixX<bool>(dijkstrastate.numNodesPerLayer, omp_thread_count()); // column major
    dijkstrastate.upperBound = std::numeric_limits<PRECISION_DIJK>::infinity();
    dijkstrastate.lowerBound = 0;
    dijkstrastate.branchId = 0;
    dijkstrastate.cost = cost.col(0);
    /*
     GRAPH CREATION
     */
    dijkstrastate.vertex2preceedingEdgeMap = buildingInGraph(numNodes + numNodesPerLayer, numNodes);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << " Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "  [ms])" << std::endl;

    /*
     FINDING INITIAL UPPER BOUND
     */


    /*
     MAIN BNB LOOP
     */
    dijkstrastate.nodeBranchId.setZero();
    MinHeap lowerBounds(numNodesPerLayer); lowerBounds.push(0.0, dijkstrastate.branchId);
    int maxBranchId = 0;
    Eigen::MatrixX<bool> branchContainsValidLowerBound(numNodesPerLayer, 1); branchContainsValidLowerBound.setZero();
    tsl::robin_map<int, std::vector<int>> validCyclesMap;
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
        const bool branchCouldContainMin = mooreBellmanFordSubroutineCPU(dijkstrastate, productspace, SRCIds, TRGTIds);
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
        const int srcIdx = dijkstrastate.cycle.at(cycleSrcIdx);
        const int trgtIdx = dijkstrastate.cycle.at(cycleTrgtIdx);
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
    return std::make_tuple(cyclePassesLayersJustOnce, dijkstrastate.upperBound, outputEdges);

}

/*
 */
DIJKSTRA_SOLVER_OUTPUT DijkstraSolver::run(const std::string& solvername) {
    const long numNodes = SRCIds.maxCoeff() + 1;
    const long numLayers = numNodes / numNodesPerLayer;

    if (solvername.compare("dijkstracpu") == 0) {
        return solveWithBnB(numLayers, numNodes);
    }
    #ifdef WITH_CUDA
    if (solvername.compare("dijkstragpu") == 0) {
        return solveWithBnBCuda(numLayers, numNodes);
    }
    #endif
    
    std::cout << "Solver >>" << solvername << "<< is not implemented" << std::endl;
    std::cout << "Falling back to >> dijkstracpu <<" << std::endl;
    return solveWithBnB(numLayers, numNodes);
}


void DijkstraSolver::setBranchGraph(const std::vector<tsl::robin_set<long>>& ibranchGraph) {
    branchGraph = ibranchGraph;
}

} // namespace dijkstra
