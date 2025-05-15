//
//  combinations.cpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#include "product_space.hpp"
#include "helper/utils.hpp"


/*function computeCombinations(...)
 Consider the edge matrix of shape X (both orientations bc triangle mesh) and the one of shape Y (single edge orientation bc 3D contour)
 -> EX is the triangle mesh
 -> EY is the contour (just a single contour, no multiple)

 -> Productspace is esentially the edges in the product graph
*/
void ProductSpace::computeCombinations() {
    const int numVerticesX = EX.maxCoeff() + 1;
    //const long numVerticesY = EY.rows();
    const long numElementsPSpace = EY.rows() * ( 2 * EX.rows() + numVerticesX);
    productspace = Eigen::MatrixXi(numElementsPSpace, 4);
    SRC_IDs  = Eigen::MatrixXi(numElementsPSpace, 1);
    TRGT_IDs = Eigen::MatrixXi(numElementsPSpace, 1);
    productspace.setZero(); SRC_IDs.setZero(); TRGT_IDs.setZero();
    piEY = Eigen::MatrixXi(EY.rows() * ( 2 * EX.rows() + numVerticesX), 2);
    piEY.setZero();

    const long numEdgesY = EY.rows();
    long numadded = 0;
    int currentstartidx = 0;
    for (int i = 0; i < numEdgesY; i++) {
        /*if (EY(i, 0) == -1) {
            productspace.row(numadded) << -1, -1, -1, -1;
            piEY(numadded, 0) = i;
            piEY(numadded, 1) = i;
            numContours++;
            currentstartidx = i+1;
            numadded++;
            continue;
        }*/
        const long target_id_base_idx = (i+1) % numEdgesY;
        for (int j = 0; j < EX.rows(); j++) {

            // intralayer
            if (!pruneIntralayerEdges) {
                productspace.row(numadded) << EY(i, 0), EY(i, 0), EX(j, 0), EX(j, 1);
                SRC_IDs(numadded)  = (int) ( i * numVerticesX + EX(j, 0) );
                TRGT_IDs(numadded) = (int) ( i * numVerticesX + EX(j, 1) );
                piEY(numadded, 0) = i;
                piEY(numadded, 1) = i;
                numadded++;
            }

            // interlayer
            productspace.row(numadded) << EY(i, 0), EY(i, 1), EX(j, 0), EX(j, 1);
            SRC_IDs(numadded)  = (int) ( i * numVerticesX + EX(j, 0) );
            TRGT_IDs(numadded) = (int) ( target_id_base_idx * numVerticesX + EX(j, 1) );
            piEY(numadded, 0) = i;
            if (i+1 < numEdgesY) {
                if (EY(i+1, 0) == -1) {
                    piEY(numadded, 1) = currentstartidx;
                }
                else {
                    piEY(numadded, 1) = i+1;
                }
            }
            else {
                piEY(numadded, 1) = currentstartidx;
            }
            numadded++;

        }

        for (int j = 0; j < numVerticesX; j++) {
            // interlayer
            productspace.row(numadded) << EY(i, 0), EY(i, 1), j, j;
            SRC_IDs(numadded)  = (int) ( i * numVerticesX + j );
            TRGT_IDs(numadded) = (int) ( target_id_base_idx * numVerticesX + j);
            piEY(numadded, 0) = i;
            if (i+1 < numEdgesY) {
                /*if (EY(i+1, 0) == -1) {
                    piEY(numadded, 1) = currentstartidx;
                }
                else {*/
                piEY(numadded, 1) = i+1;
                //}
            }
            else {
                piEY(numadded, 1) = currentstartidx;
            }
            numadded++;
        }

    }
    if (DEBUG_COMBINATIONS) std::cout << "[COMBOS] Detected " << numContours << " closed contours" << std::endl;
    productspace.conservativeResize(numadded, 4);
    SRC_IDs.conservativeResize(numadded, 1);
    TRGT_IDs.conservativeResize(numadded, 1);
    combosComputed = true;

    if (DEBUG_COMBINATIONS) {
        const int numNodesPerLayer = numVerticesX;
        for (long i = 0; i < SRC_IDs.rows(); i++) {
            const int srcId  = SRC_IDs(i);
            const int trgtId = TRGT_IDs(i);
            const Eigen::MatrixXi pedge = productspace.row(i);
            const int src2d = srcId / numNodesPerLayer;
            const int src3d = srcId - src2d * numNodesPerLayer;
            const int trgt2d = trgtId / numNodesPerLayer;
            const int trgt3d = trgtId - trgt2d * numNodesPerLayer;
            Eigen::MatrixXi pedgeReconstructed(1, 4);
            pedgeReconstructed << src2d, trgt2d, src3d, trgt3d;
            if (!utils::allEqual(pedge, pedgeReconstructed)) {
                std::cout << "Original: " << pedge << std::endl;
                std::cout << "Reconstr: " << pedgeReconstructed << std::endl;
            }
        }
    }

    if (pruneIntralayerEdges) {
        branchGraph = std::vector<tsl::robin_set<long>>();
        branchGraph.reserve(numVerticesX);
        for (int j = 0; j < numVerticesX; j++) {
            branchGraph.push_back(tsl::robin_set<long>());
        }
        for (int j = 0; j < EX.rows(); j++) {
            branchGraph.at(EX(j, 0)).insert(EX(j, 1));
        }
    }
}



ProductSpace::ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, const bool pruneIntralyer) : EX(EX), EY(EY) {
    combosComputed = false;
    numContours = 1;
    pruneIntralayerEdges = pruneIntralyer;
}

ProductSpace::ProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY) : ProductSpace(EX, EY, false) {
}


Eigen::MatrixXi ProductSpace::getProductSpace() {
    if (!combosComputed) {
        computeCombinations();
    }
    return productspace;
}

Eigen::MatrixXi ProductSpace::getPiEy() {
    if (!combosComputed) {
        computeCombinations();
    }
    return piEY;
}

Eigen::MatrixXi ProductSpace::getSRCIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return SRC_IDs;
}

Eigen::MatrixXi ProductSpace::getTRGTIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return TRGT_IDs;
}


int ProductSpace::getNumContours() const {
    return numContours;
}

std::vector<tsl::robin_set<long>> ProductSpace::getBranchGraph() {
    return branchGraph;
}
