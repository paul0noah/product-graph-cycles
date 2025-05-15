//
//  combinations.hpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#ifndef conj_combinations_hpp
#define conj_combinations_hpp

#include <Eigen/Dense>
#include "helper/shape.hpp"
#include <tsl/robin_set.h>
#define DEBUG_CONJ_COMBINATIONS false

class ConjProductSpace {
private:
    std::string prefix;
    bool combosComputed;
    Eigen::MatrixXi& EX;
    Eigen::MatrixXi& EY;
    Eigen::MatrixXi productspace;
    Eigen::MatrixXi SRC_IDs;
    Eigen::MatrixXi TRGT_IDs;
    Eigen::MatrixXi piEY;
    int numContours;
    std::tuple<int, int, std::vector<tsl::robin_set<int>>> createDataStructures();
    std::vector<tsl::robin_set<long>> branchGraph;
    bool pruneIntraLayerEdges;

public:
    void init();
    ConjProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY);
    ConjProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, bool pruneIntralayer);

    void computeCombinations();
    Eigen::MatrixXi getConjProductSpace();
    Eigen::MatrixXi getPiEy();
    Eigen::MatrixXi getSRCIds();
    Eigen::MatrixXi getTRGTIds();
    int getNumContours() const;
    std::vector<tsl::robin_set<long>> getBranchGraph();
};

#endif /* combinations_hpp */
