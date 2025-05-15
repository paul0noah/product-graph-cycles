//
//  HigherOrderDeformationEnergy.hpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 10.10.24.
//

#ifndef higherOrder_deformationEnergy_hpp
#define higherOrder_deformationEnergy_hpp
#include <Eigen/Dense>
#include "helper/shape.hpp"

#define DEBUG_HIGHER_ORDER_DEFORMATION_ENERGY true
#define USE_MATLAB_CONVENTION true

typedef enum HOD_COST_TIME_RATIO_MODE {
    NO_MODE,
    LENGTH_NORMALISATION,
    SO3_TIME_PLAIN,
    SO3_PLAIN_COST_FEAT_TIME
} HOD_COST_TIME_RATIO_MODE;


typedef enum HOD_COSTTIME_MODE {
    NOMODE,
    VANILLA,
    FEATURE,
    LENGTH,
    SO3_PLAIN,
    SO3_ROBUST,
    PENALISE_DEGENERATE
} HOD_COSTTIME_MODE;

class HigherOrderDeformationEnergy {
private:
    const Eigen::MatrixXd& VX;
    const Eigen::MatrixXi& EX;
    const Eigen::MatrixXd& NormalsX;
    const Eigen::MatrixXd& VY;
    const Eigen::MatrixXi& EY;
    const Eigen::MatrixXd& NormalsY;

    const Eigen::MatrixXi& productspace;
    const Eigen::MatrixXi& SRCIds;
    const Eigen::MatrixXi& TRGTIds;

    const Eigen::MatrixXd& featDiffMatrix;
    bool useSO3distance;
    
    bool computed;
    Eigen::MatrixXd defEnergy;
    void computeEnergy();
    std::vector<Eigen::Quaterniond> precomputeAllQuaternionds();

    double alpha;
    double c;
    double pwr;
    std::string prefix;
    double robustSO3Dif(const double x);
    bool pruneIntralyerEdges;
    std::string costName, timeName;
    HOD_COST_TIME_RATIO_MODE costTimeRatioMode;
    HOD_COSTTIME_MODE costMode;
    HOD_COSTTIME_MODE timeMode;


public:
    HigherOrderDeformationEnergy(const Eigen::MatrixXd& VX,
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
                                 const double ipwr);
    HigherOrderDeformationEnergy(const Eigen::MatrixXd& VX,
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
                                 const bool pruneIntralayer);
    Eigen::MatrixXd getEnergy();
    void setCostTimeRatioMode(const std::string icostName, const std::string itimeName);
};

#endif /* higherOrder_deformationEnergy_hpp */

