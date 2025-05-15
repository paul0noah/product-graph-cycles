//
//  DijkstraSolver_hpp.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 07.10.24
//


#ifndef DijkstraSolver_hpp
#define DijkstraSolver_hpp

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <tsl/robin_set.h>
#include <cmath>

#define DEBUG_DIJKSTRA_SOLVER false

namespace dijkstra {
typedef double PRECISION_DIJK;
typedef std::tuple<bool, float, Eigen::MatrixXi> DIJKSTRA_SOLVER_OUTPUT;
typedef std::tuple<PRECISION_DIJK, int> LBTUPLE;
bool smaller(const LBTUPLE& a, LBTUPLE& b);

struct DKSTRACPUSTATE {
    DKSTRACPUSTATE(){};
    int branchId;
    int maxDepthPerLayer;
    Eigen::MatrixXi nodeBranchId;
    Eigen::MatrixX<PRECISION_DIJK> cost;
    Eigen::MatrixX<PRECISION_DIJK> time;
    Eigen::MatrixXi predecessor;
    Eigen::MatrixX<PRECISION_DIJK> DPtable;
    Eigen::MatrixX<bool> vertexInCycleEncounter;
    std::vector<tsl::robin_set<long>> branchGraph;
    std::vector<tsl::robin_set<long>> vertex2preceedingEdgeMap;
    int numTotalNodes;
    std::string prefix;
    long numLayers;
    long numNodesPerLayer;
    long numNodes;
    PRECISION_DIJK lowerBoundOfCurrentBranch;
    PRECISION_DIJK lowerBound;
    PRECISION_DIJK upperBound;
    std::vector<int> cycle;
    std::vector<int> upperBoundCycle;
    std::vector<int> zeroLayerHits;
    int branchingSrcNode;
    int branchingTargetNode;
    int maxBranchId;
    bool upperboundFoundOnFirstTry;
    bool verbose;
    int numNodesInBranch;
    std::vector<LBTUPLE> lowerBounds;
};

class DijkstraSolver {
private:
    std::string prefix;
    int maxDepth;
    bool verbose;
    const Eigen::MatrixXi& productspace;
    const Eigen::MatrixXd& cost;
    const Eigen::MatrixXi& SRCIds;
    const Eigen::MatrixXi& TRGTIds;
    const long numNodesPerLayer;
    std::vector<tsl::robin_set<long>> branchGraph;
    bool cycleChecking;

    std::vector<tsl::robin_set<long>> buildingInGraph(const long numTotalNodes,
                                                      const long numNodes);

    DIJKSTRA_SOLVER_OUTPUT solveWithBnB(const long numLayers,
                                        const long numNodes);

#ifdef WITH_CUDA
    DIJKSTRA_SOLVER_OUTPUT solveWithBnBCuda(const long numLayers,
                                            const long numNodes);
#endif
    
public:
    DijkstraSolver(const Eigen::MatrixXi& productspace, Eigen::MatrixXd& cost, const Eigen::MatrixXi& SRCIds, const Eigen::MatrixXi& TRGTIds, const long numNodesPerLayer, const int imaxDepth);
    ~DijkstraSolver();

    void setBranchGraph(const std::vector<tsl::robin_set<long>>& branchGraph);

    DIJKSTRA_SOLVER_OUTPUT run(const std::string& solvername);// solvername = {dijkstracpu, dijkstragpu}

};

} // namespace dijkstra
#endif /* DijkstraSolver_hpp */
