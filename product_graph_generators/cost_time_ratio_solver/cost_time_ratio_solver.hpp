//
//  cost_time_ratio_solver.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 06.11.24
//


#ifndef CTRSOLVER_hpp
#define CTRSOLVER_hpp

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <tsl/robin_set.h>
#include <cmath>
namespace ctrsolver {

typedef double PRECISION_CTR;
typedef std::tuple<PRECISION_CTR, int> LBTUPLE;
bool smaller(const LBTUPLE& a, LBTUPLE& b);

typedef enum BELLMANFORDSTATUS {
    CAN_CONTAIN_OPTIMUM,
    CANNOT_CONTAIN_OPTIMUM,
    NO_NEGATIVE_CYCLE,
    FOUND_NEGATIVE_CYCLE,
    ZERO_CYCLE
} BELLMANFORDSTATUS;

struct CTRCPUSTATE {
    CTRCPUSTATE(){};
    int branchId;
    int maxDepthPerLayer;
    Eigen::MatrixXi nodeBranchId;
    Eigen::MatrixX<PRECISION_CTR> cost;
    Eigen::MatrixX<PRECISION_CTR> time;
    Eigen::MatrixXi predecessor;
    Eigen::MatrixX<PRECISION_CTR> DPtable;
    Eigen::MatrixX<bool> vertexInCycleEncounter;
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
    std::vector<LBTUPLE> lowerBounds;
    PRECISION_CTR currentRatio;
    PRECISION_CTR tolerance;
};


#define DEBUG_CRT_CYCLE_SOLVER true
typedef std::tuple<bool, float, Eigen::MatrixXi> CTR_CYCLE_SOLVER_OUTPUT;

class CostTimeRatioSolver {
private:
    std::string prefix;
    bool verbose;
    int maxDepth;
    float tolerance;
    const Eigen::MatrixXi& productspace;
    Eigen::MatrixXd& cost;
    const Eigen::MatrixXi& SRCIds;
    const Eigen::MatrixXi& TRGTIds;
    const long numNodesPerLayer;
    std::vector<tsl::robin_set<long>> branchGraph;
    bool cycleChecking;
    std::tuple<std::vector<int>, float> findInitalUpperBound(CTRCPUSTATE& ctrstate);
    bool findNegativeCylceCPU(CTRCPUSTATE& ctrstate, const bool searchForZeroCycle=false);

    CTR_CYCLE_SOLVER_OUTPUT solveWithCPULawler(const PRECISION_CTR float2intScaling,
                                               const long numLayer,
                                               const long numNodes);
    CTR_CYCLE_SOLVER_OUTPUT solveWithBnbCPULawler(const PRECISION_CTR float2intScaling,
                                                  const long numLayer,
                                                  const long numNodes);

    std::vector<tsl::robin_set<long>> buildingInGraph(const long numTotalNodes,
                                                      const long numNodes);
    std::vector<tsl::robin_set<long>> buildingOutGraph(const long numTotalNodes,
                                                       const long numNodes);
    void robustFloatToIntConversion(Eigen::MatrixXd& X, const int colIdx, const PRECISION_CTR float2intscaling);
#ifdef WITH_CUDA
    CTR_CYCLE_SOLVER_OUTPUT solveWithBnbGPULawler(const PRECISION_CTR float2intScaling,
                                                  const long numLayer,
                                                  const long numNodes);
#endif
    
public:
    CostTimeRatioSolver(const Eigen::MatrixXi& productspace, Eigen::MatrixXd& cost, const Eigen::MatrixXi& SRCIds, const Eigen::MatrixXi& TRGTIds, const long numNodesPerLayer, const int imaxDepth);
    ~CostTimeRatioSolver();

    void setBranchGraph(const std::vector<tsl::robin_set<long>>& branchGraph);

    CTR_CYCLE_SOLVER_OUTPUT run(const std::string& solvername);// solvername = {lawlercpu, }

};

} // namespace crtsolver

#endif /* CTRSOLVER_hpp */
