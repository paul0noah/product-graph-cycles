//
//  utils.cpp
//  helper
//
//  Created by Paul Rötzer on 11.04.21.
//

#include <Eigen/Dense>
#include <math.h>
#include <cmath>
#include "utils.hpp"
#include <igl/per_vertex_normals.h>
#include <igl/repmat.h>

namespace utils {

void addElement2IntVector(Eigen::VectorXi &vec, int val) {
    vec.conservativeResize(vec.rows() + 1, Eigen::NoChange);
    vec(vec.rows()-1) = val;
}

/* function safeLog
 log which is linearly extended below a threshold epsi
 */
float safeLog(const float x) {
    float l;
    if (x > FLOAT_EPSI) {
        l = std::log(x);
    }
    else {
        l = (x - FLOAT_EPSI)/FLOAT_EPSI + std::log(FLOAT_EPSI);
    }
    return l;
}

Eigen::ArrayXf arraySafeLog(const Eigen::ArrayXf X) {
    Eigen::ArrayXf L = X;
    for (int i = 0; i < X.rows(); i++) {
        L(i) = safeLog(X(i));
    }
    return L;
}

float squaredNorm(const Eigen::Vector3f vec) {
    return vec(0)*vec(0) + vec(1)*vec(1) + vec(2)*vec(2);
}


/* function setLinspaced
    creates a increasing vector of fixed step of one
    e.g.
    mat = Eigen::MatrixXi(1, 5);
    setLinspaced(mat, 2);
    creates
    mat = [2 3 4 5 6]
 */
void setLinspaced(Eigen::MatrixXi& mat, int start) {
    assert(mat.rows() == 1 || mat.cols() == 1);
    int length;
    if (mat.rows() == 1) {
        length = mat.cols();
    }
    else {
        length = mat.rows();
    }
    for (int i = 0; i < length; i++) {
        mat(i) = i + start;
    }
}

Eigen::MatrixXi linspaced(int start, int end) {
    return linspaced(start, end, 1);
}

Eigen::MatrixXi linspaced(int start, int end, int step) {
    assert(step > 0);
    assert(end > start);
    int length = (end - start)/step;
    Eigen::MatrixXi A(length, 1);
    for (int i = 0; i < length; i++) {
        A(i, 0) = start + i * step;
    }
    return A;
}

Eigen::MatrixXd normals3d(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    Eigen::MatrixXd N;
    igl::per_vertex_normals(V, F, N);
    return N;
}


Eigen::MatrixXd normals2d(Eigen::MatrixXd V, const Eigen::MatrixXi& E) {
    /*
     Setup
     */
    if (V.rows() != E.rows()) {
        std::cout << "error: V.rows() != E.rows()" << std::endl;
        return Eigen::MatrixXd(1, 1);
    }

    int zeroDimension = -1;
    for (int i = 0; i < 3; i++) {
        if (std::abs(V.col(i).sum()) < 1e-4) {
            zeroDimension = i;
            break;
        }
    }
    if (zeroDimension == -1) {
        std::cout << "error: points dont lie on xy, xz or yz plane" << std::endl;
        return Eigen::MatrixXd(1, 1);
    }
    if (zeroDimension != 2) {
        Eigen::Vector3<int> resorting;
        if (zeroDimension == 1)
            resorting << 0, 2, 1;
        else
            resorting << 1, 2, 0;

        V = V(Eigen::all, resorting);
    }
    Eigen::MatrixXd edgeVectors = V(E.col(0), Eigen::all) - V(E.col(1), Eigen::all);
    Eigen::MatrixXd edgeLenghts = rowNorm(edgeVectors);
    Eigen::MatrixXd rotationAroundZ(3, 3); rotationAroundZ.setZero();
    rotationAroundZ(0, 1) = 1;
    rotationAroundZ(1, 0) = -1;
    rotationAroundZ(2, 2) = 1;


    /*
     normal computation
     */

    Eigen::MatrixXd edgeN = edgeVectors * rotationAroundZ.transpose();

    edgeN = edgeN.rowwise().normalized();
    // check if normals point outward => bounding box must get bigger
    const float maxXVal = V.col(0).maxCoeff();
    const float minXVal = V.col(0).minCoeff();
    const float maxYVal = V.col(1).maxCoeff();
    const float minYVal = V.col(1).minCoeff();
    const float multiplier = 0.1 * std::min(std::abs(maxXVal - minXVal), std::abs(maxYVal - minYVal));
    const float maxXValmult = (V + multiplier * edgeN).col(0).maxCoeff();
    // flip normals if they dont point outwards
    if (maxXValmult < maxXVal)
        edgeN = -edgeN;

    // transform edge normals to vertex normals by a weighted sum of adjacent edge normals
    Eigen::MatrixXd vertexN(V.rows(), 3);
    for (int i = 1; i < V.rows()+1; i++) {
        const int nextIdx = i % V.rows();
        vertexN.row(nextIdx) = edgeLenghts(i-1) * edgeN.row(i-1) + edgeLenghts(nextIdx) * edgeN.row(nextIdx);
    }
    vertexN = vertexN.rowwise().normalized();

    return vertexN;
}

} // namespace utils


int findEdge(const tsl::robin_set<EDG> &ELookup, const EDG &edg) {
    auto it = ELookup.find(edg);
    if(it != ELookup.end()) {
        EDG foundEdg = *it;
        return foundEdg.e;
    }
    return -1;
}

namespace std {
    std::size_t hash<EDG>::operator()(EDG const& edg) const noexcept {
        //size_t idx0hash = std::hash<int>()(edg.idx0);
        //size_t idx1hash = std::hash<int>()(edg.idx1) << 1;
        //return idx0hash ^ idx1hash;
        int k1 = edg.idx0;
        int k2 = edg.idx1;
        return (k1 + k2 ) * (k1 + k2 + 1) / 2 + k2;
    }
}
