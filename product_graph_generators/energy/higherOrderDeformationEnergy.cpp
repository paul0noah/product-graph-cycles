//
//  HigherOrderDeformationEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 10.10.24.
//

#include "higherOrderDeformationEnergy.hpp"
#include <iostream>

#define DEBUG_SOTHREE false
#define EPSI 1e-8
#include <math.h>
#define TWO_PI 2 * M_PI

Eigen::Quaterniond precomputeRotation(const Eigen::Vector3d& v_2d,
                                      const Eigen::Vector3d& n_2d,
                                      const Eigen::Vector3d& c_2d,
                                      const Eigen::Vector3d& v_3d,
                                      const Eigen::Vector3d& n_3d,
                                      const Eigen::Vector3d& c_3d) {


    if (DEBUG_SOTHREE && v_2d.norm() < EPSI) std::cout << "v_2d zero" << std::endl;
    if (DEBUG_SOTHREE && v_3d.norm() < EPSI) std::cout << "v_3d zero" << std::endl;
    if (DEBUG_SOTHREE && n_2d.norm() < EPSI) std::cout << "n_2d zero" << std::endl;
    if (DEBUG_SOTHREE && n_3d.norm() < EPSI) std::cout << "n_3d zero" << std::endl;

    Eigen::Matrix3d X;
    X.row(0) = n_2d.normalized();
    X.row(1) = v_2d.normalized();
    X.row(2) = c_2d.normalized();

    Eigen::Matrix3d Y;
    Y.row(0) = n_3d.normalized();
    Y.row(1) = v_3d.normalized();
    Y.row(2) = c_3d.normalized();

    Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeThinU | Eigen::ComputeThinV> svd;
    svd.compute(X.transpose() * Y, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::DiagonalMatrix<double, 3> d(1, 1, (svd.matrixU() * (svd.matrixV().transpose())).determinant() );

    Eigen::Matrix3d rot1 = svd.matrixU() * d * (svd.matrixV().transpose());

    if (DEBUG_SOTHREE) {
        if ((svd.singularValues().transpose().array().abs() < EPSI).any())
            std::cout << "svd1 singular values zero" << std::endl;
        if ((X.transpose() * Y - svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose()).norm() > EPSI)
            std::cout << "SVD: A != USV^T" << std::endl;
    }

    return Eigen::Quaterniond(rot1);
}

double HigherOrderDeformationEnergy::robustSO3Dif(const double x) {
    // see https://arxiv.org/abs/1701.03077
    //const double alpha = 0.7;//-1.2;//-0.5;
    //const double c = 0.6;//0.2;//0.31831;

    const double xDivCSqr = std::pow((x / c), pwr);
    const double absAlphaMinusTwo = std::abs(alpha - 2);
    return absAlphaMinusTwo / alpha * (std::pow(xDivCSqr / absAlphaMinusTwo + 1, 0.5 * alpha) -1);
}

double getSO3DistWithPrecompute(const Eigen::Quaterniond rotation0, const Eigen::Quaterniond rotation1) {
    double innerProductRot = fabs(rotation0.w()*rotation1.w() + rotation0.vec().dot(rotation1.vec()));
    if (innerProductRot > 1) {
        innerProductRot = 1;
    }
    if (innerProductRot < -1) {
        innerProductRot = -1;
    }
    if (std::isnan(innerProductRot)) {
        return 0;
    }
    return 2 * acos(innerProductRot);
}


inline long get3dEdgeIdx(const long idxInPGraph, const long edge2dIdx, const long numNodesPerLayer, const long numEdgesX) {
    long edge3didx = idxInPGraph - edge2dIdx * numNodesPerLayer;
    if (edge3didx >= numEdgesX) {
        edge3didx = edge3didx - numEdgesX;
    }
    if (edge3didx >= numEdgesX) {
        edge3didx = edge3didx - numEdgesX;
    }
    if (edge3didx >= numEdgesX && DEBUG_HIGHER_ORDER_DEFORMATION_ENERGY) {
        std::cout << "error: edge3didx > numEdgesX, this should not happen; edge3didx=" << edge3didx << " numEdgesX=" << numEdgesX << std::endl;
    }
    return edge3didx;
}

#if USE_MATLAB_CONVENTION
std::vector<Eigen::Quaterniond> HigherOrderDeformationEnergy::precomputeAllQuaternionds() {
    if (!useSO3distance) {
        return std::vector<Eigen::Quaterniond>();
    }
    const long numEdgesX = EX.rows();
    const long numEdgesY = EY.rows();
    const long numNodesPerLayer = numEdgesX * 3;
    const long nProductNodes = numNodesPerLayer * numEdgesY;
    const long nQuaternionds = 3 * numEdgesX * numEdgesY;
    Eigen::MatrixXd edgeVecsX = VX(EX.col(1), Eigen::all) - VX(EX.col(0), Eigen::all);
    Eigen::MatrixXd edgeVecsY = VY(EY.col(1), Eigen::all) - VY(EY.col(0), Eigen::all);

    std::vector<Eigen::Quaterniond> precQuaternionds; precQuaternionds.reserve(nQuaternionds);
    for (int i = 0; i < numEdgesY; i++) { // 2d
        const Eigen::Vector3d e_2d = edgeVecsY.row(i);
        const Eigen::Vector3d n_2d = NormalsY.row(EY(i, 0));
        const Eigen::Vector3d c_2d = e_2d.cross(n_2d);
        for (int j = 0; j < numNodesPerLayer; j++) { // 3d
            const int edge3didx = j - (j / numEdgesX) * numEdgesX;
            const Eigen::Vector3d e_3d = edgeVecsX.row(edge3didx);
            const Eigen::Vector3d n_3d = (NormalsX.row(EX(edge3didx, 0)) + NormalsX.row(EX(edge3didx, 1))) / 2;
            const Eigen::Vector3d c_3d = e_3d.cross(n_3d);
            if (j < numEdgesX) { // non-deg
                precQuaternionds.push_back(precomputeRotation(e_2d, n_2d, c_2d, e_3d, n_3d, c_3d));
            }
            else if (j < numEdgesX * 2 && !pruneIntralyerEdges) { // deg-2d
                const int edge2didx = (i + numEdgesY - 1) % numEdgesY;
                const Eigen::Vector3d e_2d_ = edgeVecsY.row(edge2didx);
                const Eigen::Vector3d n_2d_ = NormalsY.row(EY(edge2didx, 0));
                const Eigen::Vector3d c_2d_ = e_2d_.cross(n_2d_);

                precQuaternionds.push_back(precomputeRotation(e_2d_, n_2d_, c_2d_, e_3d, n_3d, c_3d));
            }
            else { // deg-3d
                precQuaternionds.push_back(precomputeRotation(e_2d, n_2d, c_2d, e_3d, n_3d, c_3d));
            }
        }
    }
    return precQuaternionds;
}
#else
std::vector<Eigen::Quaterniond> HigherOrderDeformationEnergy::precomputeAllQuaternionds() {
    if (!useSO3distance) {
        return std::vector<Eigen::Quaterniond>();
    }
    const long numEdgesX = EX.rows();
    const long numEdgesY = EY.rows();
    const long numNodesPerLayer = numEdgesX * 3;
    const long nProductNodes = numNodesPerLayer * numEdgesY;
    const long nQuaternionds = numEdgesX * numEdgesY;
    Eigen::MatrixXd edgeVecsX = VX(EX.col(1), Eigen::all) - VX(EX.col(0), Eigen::all);
    Eigen::MatrixXd edgeVecsY = VY(EY.col(1), Eigen::all) - VY(EY.col(0), Eigen::all);

    std::vector<Eigen::Quaterniond> precQuaternionds; precQuaternionds.reserve(nQuaternionds);
    for (int i = 0; i < numEdgesY; i++) { // 2d
        const Eigen::Vector3d e_2d = edgeVecsY.row(i);
        const Eigen::Vector3d n_2d = NormalsY.row(EY(i, 0));
        const Eigen::Vector3d c_2d = e_2d.cross(n_2d);
        for (int j = 0; j < numEdgesX; j++) { // 3d
            const Eigen::Vector3d e_3d = edgeVecsX.row(j);
            const Eigen::Vector3d n_3d = NormalsX.row(EX(j, 0));
            const Eigen::Vector3d c_3d = e_3d.cross(n_3d);
            precQuaternionds.push_back(precomputeRotation(e_2d, n_2d, c_2d, e_3d, n_3d, c_3d));
        }
    }
    return precQuaternionds;
}
#endif


std::string modeToString(const HOD_COSTTIME_MODE mode) {
    if (mode == NOMODE)
        return "nomode";
    if (mode == VANILLA)
        return "vanilla";
    if (mode == FEATURE)
        return "feature";
    if (mode == LENGTH)
        return "length";
    if (mode == SO3_PLAIN)
        return "so3 plain";
    if (mode == SO3_ROBUST)
        return "so3 robust";
    if (mode == PENALISE_DEGENERATE)
        return "artificially penalising degenerate edges";
    return "not supported";
}

void printMode(const std::string prefix, const HOD_COSTTIME_MODE costMode, const HOD_COSTTIME_MODE timeMode) {
    std::cout << prefix << "Generating energy in mode";
    if (costMode == VANILLA && timeMode == VANILLA) {
        std::cout << ">> vanilla <<" << std::endl;;
    }
    else {
        std::string costString = modeToString(costMode);
        std::string timeString = modeToString(timeMode);
        std::cout << ">> cost = " << costString << " time = " << timeString << " <<" << std::endl;
    }
}

/*





ENERGY COMPUTATION






 */
void HigherOrderDeformationEnergy::computeEnergy() {
    bool computeLineIntegral = false;
    int numColumnsEnergy = 2;
    const float maxSO3Val = robustSO3Dif(TWO_PI);
    float maxFeatDiff;

    printMode(prefix, costMode, timeMode);
    if (timeMode == NOMODE || timeMode == VANILLA) {
        numColumnsEnergy = 1;
    }
    computeLineIntegral = costMode == LENGTH || timeMode == LENGTH;
    if (timeMode == FEATURE) {
        maxFeatDiff = featDiffMatrix.maxCoeff();
    }

    const std::vector<Eigen::Quaterniond> allquaternionds = precomputeAllQuaternionds();

    defEnergy = Eigen::MatrixXd(productspace.rows(), numColumnsEnergy);
    defEnergy.setZero();
    const long numEdgesX = EX.rows();
    const long numEdgesY = EY.rows();
    const long numNodesPerLayer = numEdgesX * 3;

    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (long i = 0; i < productspace.rows(); i++) {
        const long edge2d0 =  SRCIds(i) / numNodesPerLayer;
        const long edge2d1 = TRGTIds(i) / numNodesPerLayer;
        const long edge3d0 = get3dEdgeIdx( SRCIds(i), edge2d0, numNodesPerLayer, numEdgesX);
        const long edge3d1 = get3dEdgeIdx(TRGTIds(i), edge2d1, numNodesPerLayer, numEdgesX);
        double so3dist = 0.0;
        const long idx2d0 = productspace(i, 0);
        const long idx2d1 = productspace(i, 1);
        const long idx2d2 = productspace(i, 2);
        const long idx3d0 = productspace(i, 3);
        const long idx3d1 = productspace(i, 4);
        const long idx3d2 = productspace(i, 5);

        if (DEBUG_HIGHER_ORDER_DEFORMATION_ENERGY && idx2d0 >= VY.rows()) {
            std::cout << "idx2d0 >= VY.rows()" << idx2d0 << " " <<  VY.rows() << std::endl;
            continue;
        }
        if (DEBUG_HIGHER_ORDER_DEFORMATION_ENERGY && idx2d1 >= VY.rows()) {
            std::cout << "idx2d1 >= VY.rows()" << idx2d1 << " " <<  VY.rows() << std::endl;
            continue;
        }
        if (DEBUG_HIGHER_ORDER_DEFORMATION_ENERGY && idx3d0 >= VX.rows()) {
            std::cout << "idx3d0 >= VX.rows()" << idx3d0 << " " <<  VX.rows() << std::endl;
            continue;
        }
        if (DEBUG_HIGHER_ORDER_DEFORMATION_ENERGY && idx3d1 >= VX.rows()) {
            std::cout << "idx3d1 >= VX.rows()" << idx3d1 << " " <<  VX.rows() << std::endl;
            continue;
        }

        if (useSO3distance) {
            #if USE_MATLAB_CONVENTION
            const long quatIndex0 = SRCIds(i);
            const long quatIndex1 = TRGTIds(i);
            if (quatIndex0 >= allquaternionds.size()) { std::cout << quatIndex0 << " " << allquaternionds.size() << std::endl;}
            if (quatIndex1 >= allquaternionds.size()) { std::cout << quatIndex1 << " " << allquaternionds.size() << std::endl;}
            #else
            const long quatIndex0 = edge2d0 * numEdgesX + edge3d0;
            const long quatIndex1 = edge2d1 * numEdgesX + edge3d1;
            #endif
            so3dist = getSO3DistWithPrecompute(allquaternionds.at(quatIndex0), allquaternionds.at(quatIndex1));
        }

        const double featDiff = featDiffMatrix(idx3d2, idx2d2);//featDiffMatrix(idx3d0, idx2d0) + featDiffMatrix(idx3d1, idx2d1);
        double lineIntegralVal = 0;
        if (computeLineIntegral) {
            for (int j = 0; j < 3; j++) {
                lineIntegralVal += std::pow(VY(idx2d0, j) - VY(idx2d1, j), 2);
                //lineIntegralVal += std::pow(VY(idx2d1, j) - VY(idx2d2, j), 2);
                lineIntegralVal += std::pow(VX(idx3d0, j) - VX(idx3d1, j), 2);
                //lineIntegralVal += std::pow(VX(idx3d1, j) - VX(idx3d2, j), 2);
            }
            lineIntegralVal = std::sqrt(lineIntegralVal);
        }

        if (costMode == VANILLA && timeMode == VANILLA) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i) = so3dist  + featDiff;
        }
        else if (costMode == VANILLA && timeMode == LENGTH) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i) = (so3dist  + featDiff);
            defEnergy(i, 1) = lineIntegralVal;
        }
        else if (costMode == LENGTH && timeMode == VANILLA) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i) = lineIntegralVal;
            defEnergy(i, 1) = so3dist  + featDiff;
        }
        else if (costMode == VANILLA && timeMode == PENALISE_DEGENERATE) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i) = (so3dist  + featDiff);
            defEnergy(i, 1) = (idx2d2 == idx2d1 || idx3d2 == idx3d1) ? 0.1 : 1;
        }
        else if (costMode == PENALISE_DEGENERATE && timeMode == VANILLA) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i) = (idx2d2 == idx2d1 || idx3d2 == idx3d1) ? 0.1 : 1;
            defEnergy(i, 1) = (so3dist  + featDiff);
        }
        else if (costMode == FEATURE && timeMode == SO3_PLAIN) {
            defEnergy(i, 0) = featDiff;
            defEnergy(i, 1) = std::max(0.0, TWO_PI - so3dist); // max just to make sure no negative values
        }
        else if (costMode == FEATURE && timeMode == SO3_ROBUST) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i, 0) = featDiff;
            defEnergy(i, 1) = std::max(0.0, maxSO3Val - so3dist); // max just to make sure no negative values
        }
        else if (costMode == SO3_ROBUST && timeMode == FEATURE) {
            so3dist = robustSO3Dif(so3dist);
            defEnergy(i, 0) = so3dist;
            defEnergy(i, 1) = std::max(0.0, maxFeatDiff - featDiff);
        }
        else if (costMode == SO3_PLAIN && timeMode == FEATURE) {
            defEnergy(i, 0) = so3dist;
            defEnergy(i, 1) = std::max(0.0, maxFeatDiff - featDiff);
        }
        else {
            std::cout << prefix << "cost time mode not supported" << std::endl;
            std::vector<int> throwVec; throwVec.at(-1);
        }

    }
    computed = true;
}

HigherOrderDeformationEnergy::HigherOrderDeformationEnergy(const Eigen::MatrixXd& VX,
                                                           const Eigen::MatrixXi& EX,
                                                           const Eigen::MatrixXd& NormalsX,
                                                           const Eigen::MatrixXd& VY,
                                                           const Eigen::MatrixXi& EY,
                                                           const Eigen::MatrixXd& NormalsY,
                                                           const Eigen::MatrixXi& productspace,
                                                           const Eigen::MatrixXi& SRCIds,
                                                           const Eigen::MatrixXi& TRGTIds,
                                                           const Eigen::MatrixXd& featDiffMatrix,
                                                           const bool iuseSO3distance,
                                                           const double ialpha,
                                                           const double ic, 
                                                           const double ipwr,
                                                           const bool pruneIntralayer) :
    VX(VX), EX(EX), NormalsX(NormalsX), VY(VY), EY(EY), NormalsY(NormalsY), productspace(productspace), SRCIds(SRCIds), TRGTIds(TRGTIds), featDiffMatrix(featDiffMatrix) {
    useSO3distance = iuseSO3distance;
    computed = false;
    alpha = ialpha;
    c = ic;
    pwr = ipwr;
    prefix = "[HODefEnergy] ";
    std::cout << prefix << "Using higher order deformation energy with alpha = " << alpha << " c = " << c << " pwr = " << pwr << std::endl;
    pruneIntralyerEdges = pruneIntralayer;
    costName = "";
    timeName = "";
    costTimeRatioMode = NO_MODE;
    costMode = VANILLA;
    timeMode = VANILLA;
}

HigherOrderDeformationEnergy::HigherOrderDeformationEnergy(const Eigen::MatrixXd& VX,
                                                           const Eigen::MatrixXi& EX,
                                                           const Eigen::MatrixXd& NormalsX,
                                                           const Eigen::MatrixXd& VY,
                                                           const Eigen::MatrixXi& EY,
                                                           const Eigen::MatrixXd& NormalsY,
                                                           const Eigen::MatrixXi& productspace,
                                                           const Eigen::MatrixXi& SRCIds,
                                                           const Eigen::MatrixXi& TRGTIds,
                                                           const Eigen::MatrixXd& featDiffMatrix,
                                                           const bool iuseSO3distance,
                                                           const double ialpha,
                                                           const double ic,
                                                           const double ipwr) :
    HigherOrderDeformationEnergy(VX, EX, NormalsX, VY, EY, NormalsY, productspace, SRCIds, TRGTIds, featDiffMatrix, iuseSO3distance, ialpha, ic, ipwr, false) {
}

Eigen::MatrixXd HigherOrderDeformationEnergy::getEnergy() {
    if (!computed) {
        computeEnergy();
    }
    return defEnergy;
}

HOD_COSTTIME_MODE stringToMode(const std::string modestring) {
    if (modestring.compare("nomode") == 0)
        return NOMODE;
    if (modestring.compare("vanilla") == 0)
        return VANILLA;
    if (modestring.compare("feature") == 0)
        return FEATURE;
    if (modestring.compare("lengthnormalisation") == 0)
        return LENGTH;
    if (modestring.compare("plainso3") == 0)
        return SO3_PLAIN;
    if (modestring.compare("robustso3") == 0)
        return SO3_ROBUST;
    if (modestring.compare("penalisedegenerate"))
        return PENALISE_DEGENERATE;
    return VANILLA;
}

void HigherOrderDeformationEnergy::setCostTimeRatioMode(const std::string icostName, const std::string itimeName) {
    costName = icostName;
    timeName = itimeName;

    costMode = stringToMode(costName);
    timeMode = stringToMode(timeName);
}
