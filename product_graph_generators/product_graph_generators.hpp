//
//  product_graph_generators.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 23.05.21.
//

#ifndef ShapeMatchModel_hpp
#define ShapeMatchModel_hpp

#include "helper/shape.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include "product_graph_generators/cost_time_ratio_solver/cost_time_ratio_solver.hpp"
#include "product_graph_generators/dijkstra_solver/dijkstra_solver.hpp"

#define DEBUG_SHAPE_MATCH_MODEL true


class ProductGraphGenerators {
private:
    std::string prefix;
    bool verbose;
    Eigen::MatrixXd VX;
    Eigen::MatrixXi EX;
    Eigen::MatrixXd NormalsX;
    Eigen::MatrixXd VY;
    Eigen::MatrixXi EY;
    Eigen::MatrixXd NormalsY;
    Eigen::MatrixXd FeatDiffMatrix;

    Eigen::MatrixXi AI;
    Eigen::MatrixXi AJ;
    Eigen::MatrixXi AV;
    Eigen::MatrixXi RHS;

    Eigen::MatrixXi AIleq;
    Eigen::MatrixXi AJleq;
    Eigen::MatrixXi AVleq;
    Eigen::MatrixXi RHSleq;

    Eigen::MatrixXi productspace;
    Eigen::MatrixXi SRCIds;
    Eigen::MatrixXi TRGTIds;
    Eigen::MatrixXi piEy;
    Eigen::MatrixXd energy;
    bool modelGenerated;
    int numCouplingConstraints;
    bool regularisingCostTerm;
    int numContours;
    bool conjugateGraph;
    std::vector<tsl::robin_set<long>> branchGraph;
    bool normalsGiven;
    double rlAlpha; 
    double rlC;
    double rlPwr;
    int maxDepth;
    bool pruneIntralayerEdges;
    std::string costName, timeName;

    void writeToFile();
    
public:
    ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm);
    ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm, bool pruneIntralayer);
    ~ProductGraphGenerators();
    void generate();
    void generateConjugate();

    Eigen::MatrixXd getCostVector();
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getAVectors();
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getAleqVectors();
    void setNormals(Eigen::MatrixXd& inormalsX, Eigen::MatrixXd& inormalsY);
    void setMaxDepth(const int maxDepth);
    void setCostTimeRatioMode(const std::string costName, const std::string timeName);
    Eigen::MatrixXi getRHS();
    Eigen::MatrixXi getRHSleq();
    Eigen::MatrixXi getProductSpace();
    int getNumCouplingConstraints();
    Eigen::MatrixXi getSortedMatching(const Eigen::MatrixXi& indicatorVector);

    ctrsolver::CTR_CYCLE_SOLVER_OUTPUT solveWithCostTimeRatio(const std::string solvername);
    ctrsolver::CTR_CYCLE_SOLVER_OUTPUT solveWithCostTimeRatio(const std::string solvername, const int maxDepth);

    dijkstra::DIJKSTRA_SOLVER_OUTPUT solveWithDijkstra(const std::string solvername, const int maxDepth);
    dijkstra::DIJKSTRA_SOLVER_OUTPUT solveWithDijkstra(const std::string solvername);
    
    void updateRobustLossParams(const double alpha, const double c, const double pwr);

};

#endif /* ShapeMatchModel_hpp */
