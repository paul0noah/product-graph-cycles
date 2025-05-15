//
//  deformationEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 31.03.21.
//

#ifndef deformationEnergy_hpp
#define deformationEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"

#define DEBUG_DEFORMATION_ENERGY false


typedef enum COST_TIME_RATIO_MODE {
    MODE_NO_MODE,
    MODE_LENGTH_NORMALISATION
} COST_TIME_RATIO_MODE;

class DeformationEnergy {
private:
    Eigen::MatrixXd& VX;
    Eigen::MatrixXd& VY;

    Eigen::MatrixXi& productspace;

    Eigen::MatrixXd& featDiffMatrix;
    bool lineIntegral;
    
    bool computed;
    Eigen::MatrixXd defEnergy;
    void computeEnergy();
    std::string costName, timeName;
    COST_TIME_RATIO_MODE costTimeRatioMode;
    std::string prefix;
    
public:
    DeformationEnergy(Eigen::MatrixXd& VX, Eigen::MatrixXd& VY, Eigen::MatrixXi& productspace, Eigen::MatrixXd& featDiffMatrix, bool iLineItegral);
    Eigen::MatrixXd getDeformationEnergy();

    void setCostTimeRatioMode(const std::string icostName, const std::string itimeName);
};

#endif /* deformationEnergy_hpp */

