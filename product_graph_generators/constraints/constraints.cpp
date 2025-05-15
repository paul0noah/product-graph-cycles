#include "constraints.hpp"
#include <vector>
#include <set>

Constraints::Constraints(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, Eigen::MatrixXi& productspace, const int numContours, Eigen::MatrixXi& piEy, bool coupling, bool allowOtherSelfIntersections) :
    EX(EX), EY(EY), productspace(productspace), coupling(coupling), allowOtherSelfIntersections(allowOtherSelfIntersections), piEy(piEy), numContours(numContours) {

    nVX = EX.maxCoeff()+1;
    numCouplingConstraints = 0;
}


std::tuple<Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> Constraints::getConstraints() {
    const long nVY = EY.rows();
    Eigen::MatrixXi EY2Idx(EY.maxCoeff()+1, 1);
    EY2Idx = -EY2Idx.setOnes();
    for (int i = 0; i < nVY; i++) {
        if (EY(i, 0) == -1)
            continue;
        EY2Idx(EY(i, 0), 0) = i;
    }
    const long nEX = EX.rows();
    long nnzConstraints = nVY * (nEX + nVX) + 2 * nVY * (2 * nEX + 2 * nVX);
    long nnzLeqConstraints = 0;
    long numRowsConstraints = nVY + nVY * nVX;
    Eigen::ArrayXi VYcount;
    std::vector<std::vector<int>> couplingLayers;
    int numLayerCouples = 0;
    const long numVY = EY.maxCoeff() + 1;

    for (int i = 0; i < numVY; i++) {
        couplingLayers.push_back(std::vector<int>());
    }
    // find double or triple vertices
    VYcount = Eigen::ArrayXi(numVY, 1);
    VYcount.setZero();
    if (coupling) {
        int nthcontour = 0;
        for (int i = 0; i < EY.rows(); i++) {
            if (EY(i, 0) == -1) {
                nthcontour++;
                continue;
            }
            const int vertexyIdx = EY(i, 0);
            VYcount(vertexyIdx, 0) = VYcount(vertexyIdx, 0) + 1;
            couplingLayers[vertexyIdx].push_back(i - nthcontour); // we might need to substract -1 here, ✅
        }
        for (int i = 0; i < numVY; i++) {
            if (VYcount(i, 0) > 1) {
                numLayerCouples += VYcount(i, 0) - 1;
            }
        }

        if (numLayerCouples == 0) {
            if (DEBUG_CONSTRAINTS) std::cout << "[CONSTR] disabling coupling bc no intersections found" << std::endl;
            // no intersections found => no coupling constraints
            coupling = false;
        }
        else {
            numCouplingConstraints = numLayerCouples;
            if (DEBUG_CONSTRAINTS) std::cout << "[CONSTR] found "<< numLayerCouples << " coupling constraints" << std::endl;
            nnzConstraints += numLayerCouples * 2 * (nEX * 2 + nVX);
            numRowsConstraints += numLayerCouples * nVX;
        }
    }

    if (!allowOtherSelfIntersections) {
        nnzLeqConstraints = (nVY - numLayerCouples) * (2 * nEX + nVX);
    }

    Eigen::MatrixXi I(nnzConstraints, 1); I.setZero();
    Eigen::MatrixXi J(nnzConstraints, 1); J.setZero();
    Eigen::MatrixXi V(nnzConstraints, 1); V.setZero();
    Eigen::MatrixXi Ileq(nnzLeqConstraints, 1); Ileq.setZero();
    Eigen::MatrixXi Jleq(nnzLeqConstraints, 1); Jleq.setZero();
    Eigen::MatrixXi Vleq(nnzLeqConstraints, 1); Vleq.setZero();
    long idx = 0;

    Eigen::MatrixXi RHS(numRowsConstraints, 1);
    Eigen::MatrixXi RHSleq(nVX, 1);
    RHS.setZero();

    long offset = 0;
    // projection on inter layer edges for each layer has to be == 1 (for cycle)
    std::set<int> zerorows;
    for (long i = 0; i < productspace.rows(); i++) {
        if (productspace(i, 0) == -1) {
            zerorows.insert(piEy(i));
            continue;
        }
        const bool interLayerEdge = productspace(i, 0) != productspace(i, 1);
        if (interLayerEdge) {
            const int rowIdx = piEy(i);//;EY2Idx(productspace(i, 0), 0);
            if (DEBUG_CONSTRAINTS && EY2Idx(productspace(i, 0), 0) < 0) {
                std::cout << "[CONSTR] EY2Idx(productspace(i, 0), 0) < 0 first loop" << std::endl;
            }

            I(idx, 0) = rowIdx;
            J(idx, 0) = (int) i;
            V(idx, 0) = 1;
            idx++;
            if (rowIdx > offset) {
                offset = rowIdx;
            }
        }

        if (DEBUG_CONSTRAINTS && idx >= nnzConstraints) {
            std::cout << "[CONSTR] idx beyond expected nonzeros in first loop" << std::endl;
            break;
        }
    }
    for (long i = 0; i < offset + 1; i++) {
        RHS(i, 0) = 1;
    }
    for (const int &row : zerorows) {
        RHS(row, 0) = 0; // we have to set the weird rows to zero not one otherwise the model is infeasible
    }
    // the rest of RHS is zero i guess


    const long offset2 = offset + 1;
    // in and out of each node has to be 0 constraint
    for (long i = 0; i < productspace.rows(); i++) {
        if (productspace(i, 0) == -1) {
            continue;
        }
        //const long inNodeIdx = EY2Idx(productspace(i, 1), 0) * nVX + productspace(i, 3); // we go into this node
        const long inNodeIdx = piEy(i, 0) * nVX + productspace(i, 2); // we go into this node
        if (DEBUG_CONSTRAINTS && EY2Idx(productspace(i, 1), 0) < 0) {
            std::cout << "[CONSTR] EY2Idx(productspace(i, 1), 0) < 0 second loop inNodeIdx" << std::endl;
        }
        //const long outNodeIdx = EY2Idx(productspace(i, 0), 0) * nVX + productspace(i, 2); // we leave this node
        const long outNodeIdx = piEy(i, 1) * nVX + productspace(i, 3); // we leave this node
        if (DEBUG_CONSTRAINTS && EY2Idx(productspace(i, 0), 0) < 0) {
            std::cout << "[CONSTR] EY2Idx(productspace(i, 0), 0) < 0 second loop outNodeIdx" << std::endl;
        }

        if (DEBUG_CONSTRAINTS) {
            if (inNodeIdx >= (nVY + 1) * nVX) {
                std::cout << "[CONSTR] inNodeIdx  >= nVY * nVX " << inNodeIdx << " productspace(i, 1), productspace(i, 3) "
                          << productspace(i, 1) << ", " << productspace(i, 3)<<  std::endl;
                continue;
            }
            if (outNodeIdx >= (nVY + 1) * nVX) {
                std::cout << "[CONSTR] outNodeIdx  >= nVY * nVX " << outNodeIdx << " productspace(i, 0), productspace(i, 2) "
                          << productspace(i, 0) << ", " << productspace(i, 2)<<  std::endl;
                continue;
            }
        }

        // in means +1
        I(idx, 0) = (int) (offset2 + inNodeIdx);
        J(idx, 0) = (int) i;
        V(idx, 0) = 1;
        idx++;

        // out means -1
        I(idx, 0) = (int) (offset2 + outNodeIdx);
        J(idx, 0) = (int) i;
        V(idx, 0) = -1;
        idx++;


        if (offset2 + inNodeIdx > offset) {
            offset = offset2 + inNodeIdx;;
        }

        if (offset2 + outNodeIdx > offset) {
            offset = offset2 + outNodeIdx;;
        }

        if (DEBUG_CONSTRAINTS && idx >= nnzConstraints) {
            std::cout << "[CONSTR] idx beyond expected nonzeros in second loop i="<<i << std::endl;
            break;
        }
    }

    if (DEBUG_CONSTRAINTS && offset + 1 - offset2 != nVY * nVX) {
        std::cout << "[CONSTR] offset - offset2 != nVY * nVX, offset - offset2 = "<< offset - offset2 << ", " << nVY * nVX << std::endl;
    }

    const long offset3 = offset + 1;
    // coupling of layers if they resemble same Y vertex and coupling is enabled
    if (coupling) {
        const long numEdgesOnLayer = (2 * nEX + nVX);
        int nthCoupling = 0;
        for (long i = 0; i < numVY; i++) {
            if (VYcount(i, 0) > 1) {
                int baseCoupleLayer = -1;
                try {
                    auto& iiiii = couplingLayers.at(i);
                }
                catch (const std::exception &exc) {
                    std::cout << " i = " << i << std::endl;
                    std::cout << " couplingLayers.size() = " << couplingLayers.size() << std::endl;
                    std::cout << exc.what() << std::endl;
                }

                for (auto& layer : couplingLayers.at(i)) { // layer is currently index INCLUDING -1 rows
                    if (baseCoupleLayer == -1) {
                        baseCoupleLayer = layer;
                        continue;
                    }
                    int nthcontour = 0;
                    /*for (int j = 0; j < layer; j++) {
                        if (EY(j, 0) == -1) {
                            nthcontour++;
                        }
                    }*/


                    // any outgoing edge of vertex in X in layer i must be matched by any outgoing edge in baseCoupleLayer
                    const long edgeOffsetBase = baseCoupleLayer * numEdgesOnLayer; // we also need to fix this here!!!!
                    const long edgeOffsetCouple = layer * numEdgesOnLayer;
                    if (edgeOffsetCouple > productspace.rows() - numEdgesOnLayer) {
                        std::cout << " nthcontour" << nthcontour << std::endl;
                    }
                    for (long j = 0; j < numEdgesOnLayer; j++) {
                        const long rowIdx = nthCoupling * nVX + productspace(j, 2); // actually only on 0-th layer but should be fine

                        // base layer +1
                        I(idx, 0) = (int) (offset3 + rowIdx);
                        J(idx, 0) = (int) (edgeOffsetBase + j);
                        V(idx, 0) = 1;
                        idx++;

                        // coupled layer -1
                        I(idx, 0) = (int) (offset3 + rowIdx);
                        J(idx, 0) = (int) (edgeOffsetCouple + j);
                        V(idx, 0) = -1;
                        idx++;

                    }

                    nthCoupling++;
                }

                if (DEBUG_CONSTRAINTS && idx >= nnzConstraints) {
                    std::cout << "[CONSTR] idx beyond expected nonzeros in third loop" << std::endl;
                    break;
                }
            }
        }

    }

    if (!allowOtherSelfIntersections) {
        long numAddedLeq = 0;
        RHSleq.setOnes();

        for (long i = 0; i < productspace.rows(); i++) {
            // if degenerate 3d edge we do not add this to the constraint since we can "stay in the 3d vertex for multiple consecutive layers"
            if (productspace(i, 3) == productspace(i, 2)) {
                continue;
            }
            if (productspace(i, 0) == -1) {
                continue;
            }

            //const long inNodeIdx = piEy(i, 0) * nVX + productspace(i, 2); // we go into this node

            const long outNodeIdx = productspace(i, 3);//piEy(i, 1) * nVX + productspace(i, 3); // we leave this node


            /*/ in
            if (VYcount(productspace(i, 0), 0)  <= 1) {
                Ileq(numAddedLeq, 0) = (int) (inNodeIdx);
                Jleq(numAddedLeq, 0) = (int) i;
                Vleq(numAddedLeq, 0) = 1;
                numAddedLeq++;

                if (numAddedLeq >= nnzLeqConstraints) {
                    std::cout << "[CONSTR] numAddedLeq >= nnzLeqConstraints " << numAddedLeq << " " << nnzLeqConstraints << std::endl;
                    break;
                }
            }*/

            // out
            if (VYcount(productspace(i, 1), 0) <= 1) {
                Ileq(numAddedLeq, 0) = (int) outNodeIdx;//(outNodeIdx + nVY * nVX);
                Jleq(numAddedLeq, 0) = (int) i;
                Vleq(numAddedLeq, 0) = 1;
                numAddedLeq++;

                if (numAddedLeq > nnzLeqConstraints) {
                    std::cout << "[CONSTR] numAddedLeq > nnzLeqConstraints " << numAddedLeq << " " << nnzLeqConstraints << std::endl;
                    break;
                }
            }


        }
        Ileq.conservativeResize(numAddedLeq, 1);
        Jleq.conservativeResize(numAddedLeq, 1);
        Vleq.conservativeResize(numAddedLeq, 1);
    }

    I.conservativeResize(idx, 1);
    J.conservativeResize(idx, 1);
    V.conservativeResize(idx, 1);
    return std::make_tuple(I, J, V, RHS, Ileq, Jleq, Vleq, RHSleq);
}



int Constraints::getNumCouplingConstr() {
    return numCouplingConstraints;
}
