//
//  product_graph_generators.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 23.05.21.
//

#include "product_graph_generators.hpp"
#include <fstream>
#include <iostream>
#include <cstdio>
#include "helper/utils.hpp"
#include <chrono>
#include <filesystem>
#include <algorithm>
#include "product_graph_generators/product_spaces/conj_product_space.hpp"
#include "product_graph_generators/energy/deformationEnergy.hpp"
#include "product_graph_generators/energy/higherOrderDeformationEnergy.hpp"
#include "product_graph_generators/constraints/constraints.hpp"

void ProductGraphGenerators::generateConjugate() {
    if (verbose && !normalsGiven) {
        std::cout << prefix << "error: cannot generate Shape Match Model without normals provided by user" << std::endl;
        return;
    }

    if (verbose) std::cout << prefix << "Generating Shape Match Model for conjugate product space..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    const std::string pruneOutput = pruneIntralayerEdges ? " (pruning intralayer edges)" : "";
    if (verbose) std::cout << prefix << "  > Product Space" << pruneOutput << std::endl;
    ConjProductSpace combos(EX, EY, pruneIntralayerEdges);
    productspace = combos.getConjProductSpace();
    numContours = combos.getNumContours();
    SRCIds = combos.getSRCIds();
    TRGTIds = combos.getTRGTIds();
    branchGraph = combos.getBranchGraph();

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "  [ms])" << std::endl;
    if (verbose) std::cout << prefix << "  > Energies" << std::endl;
    HigherOrderDeformationEnergy defEnergy(VX, EX, NormalsX, VY, EY, NormalsY, productspace, SRCIds, TRGTIds, FeatDiffMatrix, regularisingCostTerm, rlAlpha, rlC, rlPwr, pruneIntralayerEdges);
    defEnergy.setCostTimeRatioMode(costName, timeName);
    energy = defEnergy.getEnergy();

    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "  [ms])" << std::endl;
    if (verbose) std::cout << prefix << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() << "  [ms])" << std::endl;

    modelGenerated = true;
}


void ProductGraphGenerators::generate() {
    if (conjugateGraph) return generateConjugate();


    if (verbose) std::cout << prefix << "Generating Shape Match Model for normal product space..." << std::endl;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    const std::string pruneOutput = pruneIntralayerEdges ? " (pruning intralayer edges)" : "";
    if (verbose) std::cout << prefix << "  > Product Space" << pruneOutput << pruneOutput << std::endl;
    ProductSpace combos(EX, EY, pruneIntralayerEdges);
    productspace = combos.getProductSpace();
    numContours = combos.getNumContours();
    SRCIds = combos.getSRCIds();
    TRGTIds = combos.getTRGTIds();
    if (pruneIntralayerEdges) {
        branchGraph = combos.getBranchGraph();
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "  [ms])" << std::endl;
    
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "  [ms])" << std::endl;
    if (verbose) std::cout << prefix << "  > Energies" << std::endl;
    DeformationEnergy defEnergy(VX, VY, productspace, FeatDiffMatrix, regularisingCostTerm);
    defEnergy.setCostTimeRatioMode(costName, timeName);

    energy = defEnergy.getDeformationEnergy();
    
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    if (verbose) std::cout << prefix << "  Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << "  [ms])" << std::endl;
    if (verbose) std::cout << prefix << "Done (" << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t1).count() << "  [ms])" << std::endl;

    modelGenerated = true;
}



ProductGraphGenerators::ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm, bool pruneIntralayer) {
    prefix = "[ShapeMM] ";
    VX = iVX;
    EX = iEX;
    VY = iVY;
    EY = iEY;
    FeatDiffMatrix = iFeatDiffMatrix;
    modelGenerated = false;
    verbose = true;
    numCouplingConstraints = 0;
    conjugateGraph = iConjugateGraph;
    regularisingCostTerm = iRegularisingCostTerm;
    numContours = 0;
    rlAlpha = 0.7; 
    rlC = 0.6;
    rlPwr = 4;
    maxDepth = 4;
    pruneIntralayerEdges = pruneIntralayer;
    costName = "vanilla";
    timeName = "";
}

ProductGraphGenerators::ProductGraphGenerators(Eigen::MatrixXd& iVX, Eigen::MatrixXi& iEX, Eigen::MatrixXd& iVY, Eigen::MatrixXi& iEY, Eigen::MatrixXd& iFeatDiffMatrix, bool iConjugateGraph, bool iRegularisingCostTerm) : ProductGraphGenerators(iVX, iEX, iVY, iEY, iFeatDiffMatrix, iConjugateGraph, iRegularisingCostTerm, false) {
}

void ProductGraphGenerators::setNormals(Eigen::MatrixXd& inormalsX, Eigen::MatrixXd& inormalsY){
    if (inormalsX.rows() != VX.rows() || inormalsX.cols() != 3) {
        std::cout << prefix << "error: normals for shape X do not contain as many entries as expected. Assumed shape = |VX| x 3" << std::endl;
        return;
    }
    if (inormalsY.rows() != VY.rows() || inormalsY.cols() != 3) {
        std::cout << prefix << "error: normals for shape Y do not contain as many entries as expected. Assumed shape = |VY| x 3" << std::endl;
        return;
    }
    NormalsX = inormalsX;
    NormalsY = inormalsY;
    normalsGiven = true;
}


ProductGraphGenerators::~ProductGraphGenerators() {
    
}


Eigen::MatrixXd ProductGraphGenerators::getCostVector() {
    if (!modelGenerated) {
        generate();
    }
    return energy;
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> ProductGraphGenerators::getAVectors() {
    if (!modelGenerated) {
        generate();
    }
    return std::make_tuple(AI, AJ, AV);
}

Eigen::MatrixXi ProductGraphGenerators::getRHS() {
    if (!modelGenerated) {
        generate();
    }
    return RHS;
}

std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> ProductGraphGenerators::getAleqVectors() {
    if (!modelGenerated) {
        generate();
    }
    return std::make_tuple(AIleq, AJleq, AVleq);
}

Eigen::MatrixXi ProductGraphGenerators::getRHSleq() {
    if (!modelGenerated) {
        generate();
    }
    return RHSleq;
}

Eigen::MatrixXi ProductGraphGenerators::getProductSpace() {
    if (!modelGenerated) {
        generate();
    }
    return productspace;
}

int ProductGraphGenerators::getNumCouplingConstraints() {
    return numCouplingConstraints;
}

Eigen::MatrixXi ProductGraphGenerators::getSortedMatching(const Eigen::MatrixXi& indicatorVector) {
    const long maxNumEdgesOnLevel = productspace.rows() / EY.rows();


    const long numCycleConstr = EY.rows();
    const long numInOutConstr = EY.rows() * VX.rows();
    std::vector<Eigen::Triplet<int>> in_out_entries;
    in_out_entries.reserve(AI.rows());
    for (long i = 0; i < AI.rows(); i++) {
        if (AI(i) >= numCycleConstr && AI(i) < numCycleConstr + numInOutConstr) {
            in_out_entries.push_back(Eigen::Triplet<int>(AI(i) - (int) numCycleConstr, AJ(i), AV(i)));
        }
    }

    Eigen::SparseMatrix<int, Eigen::RowMajor> in_out_rm(numInOutConstr, productspace.rows());
    in_out_rm.setFromTriplets(in_out_entries.begin(), in_out_entries.end());
    Eigen::SparseMatrix<int, Eigen::ColMajor> in_out_cm(numInOutConstr, productspace.rows());
    in_out_cm.setFromTriplets(in_out_entries.begin(), in_out_entries.end());


    long firstNonZeroIdx = -1;
    long numMatches = 0;
    for (long i = 0; i < indicatorVector.rows(); i++) {
        if (indicatorVector(i) == 1) {
            numMatches++;
            if (firstNonZeroIdx == -1)
                firstNonZeroIdx = i;
        }
    }
    Eigen::MatrixXi matchingSorted(numMatches, 4);
    matchingSorted = -matchingSorted.setOnes();
    Eigen::MatrixXi nodeUsed(numInOutConstr, 1); nodeUsed.setZero();
    //Eigen::MatrixXi sortedIndices(numMatches, 1); sortedIndices = -sortedIndices.setOnes();

    long idx = firstNonZeroIdx;
    for (long i = 0; i < numMatches; i++) {
        //sortedIndices(i, 1) = (int) idx;
        //std::cout << idx << ": " <<productspace.row(idx) << std::endl;
        matchingSorted.row(i) = productspace.row(idx);
        long row = -1;
        int currentVal = 0;
        bool newNodeFound = false;
        for (typename Eigen::SparseMatrix<int, Eigen::ColMajor>::InnerIterator it(in_out_cm, idx); it; ++it) {
            if (nodeUsed(it.row(), 0) == 0 && it.value() == -1) {
                //std::cout << "  " << it.value() << std::endl;
                row = it.row();
                nodeUsed(row, 0) = 1;
                currentVal = it.value();
                newNodeFound = true;
                break;
            }
        }
        if (!newNodeFound) {
            if (DEBUG_SHAPE_MATCH_MODEL) std::cout << prefix << "Did not find new node, aborting" << std::endl;
            long numadded = 0;
            for (long ii = 0; ii < indicatorVector.rows(); ii++) {
                if (indicatorVector(ii) == 1) {
                    matchingSorted.row(numadded) = productspace.row(ii);
                    numadded++;
                }
            }
            break;
        }

        for (typename Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(in_out_rm, row); it; ++it) {
            if (it.value() == -currentVal && indicatorVector(it.col(), 0) > 0) {
                idx = it.col();
                break;
            }
        }
    }

    return matchingSorted;
}

void ProductGraphGenerators::updateRobustLossParams(const double alpha, const double c, const double pwr) {
    if (!conjugateGraph) {
        return;
    }
    if (modelGenerated) {
        std::cout << prefix << "error: cannot update robust loss params after model has been generated" << std::endl;
        return;
    }
    rlAlpha = alpha; 
    rlC = c;
    rlPwr = pwr;
}

void ProductGraphGenerators::writeToFile() {
    utils::writeMatrixToFile(VX, "VX");
    utils::writeMatrixToFile(VY, "VY");
    utils::writeMatrixToFile(EX, "EX");
    utils::writeMatrixToFile(EY, "EY");
    utils::writeMatrixToFile(FeatDiffMatrix, "FeatDiffMatrix");
    if (conjugateGraph) {
        utils::writeMatrixToFile(NormalsX, "NX");
        utils::writeMatrixToFile(NormalsY, "NY");
    }
}

ctrsolver::CTR_CYCLE_SOLVER_OUTPUT ProductGraphGenerators::solveWithCostTimeRatio(const std::string solvername, const int maxDepth) {
    if (solvername.compare("export") == 0) {
        writeToFile();
        return std::make_tuple(false, -1.0, Eigen::MatrixXi(4, 1));
    }
    if (!modelGenerated) {
        generate();
    }
    using namespace ctrsolver;
    const long numNodesPerLayer = conjugateGraph ? EX.rows() * 3 : VX.rows();

    CostTimeRatioSolver ctrsolver = CostTimeRatioSolver(productspace, energy, SRCIds, TRGTIds, numNodesPerLayer, maxDepth);
    if (conjugateGraph || pruneIntralayerEdges) ctrsolver.setBranchGraph(branchGraph);
    return ctrsolver.run(solvername);
}


ctrsolver::CTR_CYCLE_SOLVER_OUTPUT ProductGraphGenerators::solveWithCostTimeRatio(const std::string solvername) {
    return solveWithCostTimeRatio(solvername, maxDepth);
}

void ProductGraphGenerators::setMaxDepth(const int imaxDepth) {
    maxDepth = imaxDepth;
}

void ProductGraphGenerators::setCostTimeRatioMode(const std::string icostName, const std::string itimeName) {
    costName = icostName;
    timeName = itimeName;
}

dijkstra::DIJKSTRA_SOLVER_OUTPUT ProductGraphGenerators::solveWithDijkstra(const std::string solvername, const int maxDepth) {
    if (solvername.compare("export") == 0) {
        writeToFile();
        return std::make_tuple(false, -1.0, Eigen::MatrixXi(4, 1));
    }
    if (!modelGenerated) {
        generate();
    }
    using namespace dijkstra;
    const long numNodesPerLayer = conjugateGraph ? EX.rows() * 3 : VX.rows();

    DijkstraSolver dijkstrasolver = DijkstraSolver(productspace, energy, SRCIds, TRGTIds, numNodesPerLayer, maxDepth);
    if (conjugateGraph || pruneIntralayerEdges) dijkstrasolver.setBranchGraph(branchGraph);
    return dijkstrasolver.run(solvername);
}


ctrsolver::CTR_CYCLE_SOLVER_OUTPUT ProductGraphGenerators::solveWithDijkstra(const std::string solvername) {
    return solveWithDijkstra(solvername, maxDepth);
}
