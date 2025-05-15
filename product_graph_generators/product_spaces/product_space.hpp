//
//  combinations.hpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#ifndef combinations_hpp
#define combinations_hpp

#include <Eigen/Dense>
#include "helper/shape.hpp"
#define DEBUG_COMBINATIONS false

class ProductSpace {
private:
    bool combosComputed;
    Eigen::MatrixXi& EX;
    Eigen::MatrixXi& EY;
    Eigen::MatrixXi productspace;
    Eigen::MatrixXi SRC_IDs;
    Eigen::MatrixXi TRGT_IDs;
    Eigen::MatrixXi piEY;
    int numContours;
    bool pruneIntralayerEdges;
    std::vector<tsl::robin_set<long>> branchGraph;

public:
    void init();
    ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY);
    ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, const bool pruneIntralyer);
    
    void computeCombinations();
    Eigen::MatrixXi getProductSpace();
    Eigen::MatrixXi getPiEy();
    Eigen::MatrixXi getSRCIds();
    Eigen::MatrixXi getTRGTIds();
    int getNumContours() const;
    std::vector<tsl::robin_set<long>> getBranchGraph();
};

#endif /* combinations_hpp */
