//
//  deformationEnergy.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#include "deformationEnergy.hpp"
#include <iostream>

void DeformationEnergy::computeEnergy() {
    int numColumnsEnergy = 2;
    std::cout << prefix << "Generating energy in mode ";
    if (costTimeRatioMode == MODE_NO_MODE) {
        std::cout << ">> vanilla <<" << std::endl;;
        numColumnsEnergy = 1;
    }
    else if (costTimeRatioMode == MODE_LENGTH_NORMALISATION) {
        std::cout << ">> length normalisation <<" << std::endl;;
        lineIntegral = true;
    }

    defEnergy = Eigen::MatrixXd(productspace.rows(), numColumnsEnergy);
    defEnergy.setZero();

    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (long i = 0; i < productspace.rows(); i++) {
        double lineIntegralVal = 1.0;
        const long idx2d0 = productspace(i, 0);
        const long idx2d1 = productspace(i, 1);
        const long idx3d0 = productspace(i, 2);
        const long idx3d1 = productspace(i, 3);

        if (idx2d0 == -1) {
            // handle weird inbetween elements
            defEnergy(i, 0) = 999999999.9;
            continue;
        }

        if (DEBUG_DEFORMATION_ENERGY && idx2d0 >= VY.rows()) {
            std::cout << "idx2d0 >= VY.rows()" << idx2d0 << " " <<  VY.rows() << std::endl;
            continue;
        }
        if (DEBUG_DEFORMATION_ENERGY && idx2d1 >= VY.rows()) {
            std::cout << "idx2d1 >= VY.rows()" << idx2d1 << " " <<  VY.rows() << std::endl;
            continue;
        }
        if (DEBUG_DEFORMATION_ENERGY && idx3d0 >= VX.rows()) {
            std::cout << "idx3d0 >= VX.rows()" << idx3d0 << " " <<  VX.rows() << std::endl;
            continue;
        }
        if (DEBUG_DEFORMATION_ENERGY && idx3d1 >= VX.rows()) {
            std::cout << "idx3d1 >= VX.rows()" << idx3d1 << " " <<  VX.rows() << std::endl;
            continue;
        }

        if (lineIntegral) {
            lineIntegralVal = 0;
            for (int j = 0; j < 3; j++) {
                lineIntegralVal += std::pow(VY(idx2d0, j) - VY(idx2d1, j), 2);
                lineIntegralVal += std::pow(VX(idx3d0, j) - VX(idx3d1, j), 2);
            }
            lineIntegralVal = std::sqrt(lineIntegralVal);
        }
        const double featDiff = featDiffMatrix(idx3d0, idx2d0) + featDiffMatrix(idx3d1, idx2d1);
        if (costTimeRatioMode == MODE_NO_MODE) {
            defEnergy(i, 0) = 0.5 * featDiff * lineIntegralVal;
        }
        else if (costTimeRatioMode == MODE_LENGTH_NORMALISATION) {
            defEnergy(i, 0) = featDiff;
            defEnergy(i, 1) = lineIntegralVal;
        }

    }
    computed = true;
}

DeformationEnergy::DeformationEnergy(Eigen::MatrixXd& VX, Eigen::MatrixXd& VY, Eigen::MatrixXi& productspace, Eigen::MatrixXd& featDiffMatrix, bool iLineItegral) :
    VX(VX), VY(VY), productspace(productspace), featDiffMatrix(featDiffMatrix) {
    lineIntegral = iLineItegral;
    computed = false;
    costName = "";
    timeName = "";
    prefix = "[DefEnergy] ";
    costTimeRatioMode = MODE_NO_MODE;
}

Eigen::MatrixXd DeformationEnergy::getDeformationEnergy() {
    if (!computed) {
        computeEnergy();
    }
    return defEnergy;
}

void DeformationEnergy::setCostTimeRatioMode(const std::string icostName, const std::string itimeName) {
    costName = icostName;
    timeName = itimeName;
    if (costName.compare("feature") == 0 && timeName.compare("lengthnormalisation") == 0) {
        costTimeRatioMode = MODE_LENGTH_NORMALISATION;
    }
}
