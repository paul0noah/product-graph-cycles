//
//  constraints.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 17.04.21.
//

#ifndef constraints_hpp
#define constraints_hpp

#include <Eigen/Sparse>
#include "helper/shape.hpp"
#include "product_graph_generators/product_spaces/product_space.hpp"
#include "helper/utils.hpp"

#define DEBUG_CONSTRAINTS true


class Constraints {
private:
    Eigen::MatrixXi& EX;
    Eigen::MatrixXi& EY;
    Eigen::MatrixXi& productspace;
    Eigen::MatrixXi& piEy;
    bool coupling;
    bool allowOtherSelfIntersections;
    long nVX;
    int numCouplingConstraints;
    int numContours;

public:
    Constraints(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, Eigen::MatrixXi& productspace, const int numContours, Eigen::MatrixXi& piEy, bool coupling, bool allowOtherSelfIntersections);
    std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> getConstraints();
    int getNumCouplingConstr();
};
#endif /* constraints_hpp */
