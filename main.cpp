#include <iostream>
#include <product_graph_generators/product_graph_generators.hpp>

Shape getTestShapeBase() {
    Eigen::MatrixXf V(8, 3);
    V <<
    0.018735, -0.181265, 0.001873,
    0.018735,  1.243550, 0.001873,
    1.206080,  0.056204, 0.020608,
    1.206080,  1.243550, 0.020608,
    0.018735,  0.056204, 1.020608,
    0.018735,  1.243550, 1.020608,
    1.206080,  0.056204, 1.226688,
    1.206080,  1.243550, 1.226688;
    Eigen::MatrixXi F(12, 3);
    F <<
        0, 1, 2,
        0, 2, 6,
        0, 4, 1,
        0, 6, 4,
        1, 3, 2,
        1, 4, 5,
        1, 5, 3,
        2, 3, 6,
        3, 5, 7,
        3, 7, 6,
        4, 6, 5,
        5, 6, 7;

    Shape testshape = Shape(V, F);
    return testshape;
}

int main() {
    Shape sx = getTestShapeBase();
    Eigen::MatrixXd VX = sx.getV().cast<double>();
    Eigen::MatrixXi EX = sx.getE();

    const int numVertY = 10;
    Eigen::MatrixXd VY = Eigen::MatrixXd(numVertY, 3);
    Eigen::MatrixXi EY = Eigen::MatrixXi(numVertY, 2);
    for (int i = 0; i < numVertY; i++) {
        EY.row(i) << i, (i+1)%numVertY;
    }

    Eigen::MatrixXd featDiff(VX.rows(), numVertY);
    for (int row = 0; row < VX.rows(); row++) {
        for (int col = 0; col < numVertY; col++) {
            featDiff(row, col) = row * VX.rows() + col + 1.234;
        }
    }
    featDiff = featDiff.setRandom().cwiseAbs();

    ProductGraphGenerators pgen = ProductGraphGenerators(VX, EX, VY, EY, featDiff, false, false);
    pgen.setMaxDepth(2);
    pgen.solveWithCostTimeRatio("lawlercpu");

}
