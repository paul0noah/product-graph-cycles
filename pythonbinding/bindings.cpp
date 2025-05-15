//
//  ShapeMatchModelPyBinds.cpp
//  dual-decompositions
//
//  Created by Paul Rötzer on 27.06.22.
//

#include "product_graph_generators/product_graph_generators.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "helper/utils.hpp"


namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT

void notimplemented() {
    std::cout << "TODO: IMPL" << std::endl;
}

PYBIND11_MODULE(cyclic_product_graphs, handle) {
    handle.doc() = "Python bindings for Higher-Order Ratio Cycles for Fast and Globally Optimal Shape Matching";

    // helper functions
    handle.def("compute_normals_3d", &utils::normals3d);
    handle.def("compute_normals_2d", &utils::normals2d);
    handle.def("compute_local_thickness_3d", &notimplemented);
    handle.def("compute_local_thickness_2d", &notimplemented);


    py::class_<ProductGraphGenerators, std::shared_ptr<ProductGraphGenerators>> smm(handle, "product_graph_generator");
    smm.def(py::init<Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&, bool, bool>());
    smm.def(py::init<Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&, bool, bool, bool>());
    smm.def("generate", &ProductGraphGenerators::generate);
    smm.def("get_cost_vector", &ProductGraphGenerators::getCostVector);
    smm.def("update_robust_loss_params", &ProductGraphGenerators::updateRobustLossParams);
    smm.def("set_normals", &ProductGraphGenerators::setNormals);
    smm.def("get_product_space", &ProductGraphGenerators::getProductSpace);

    smm.def("solve_with_cost_time_ratio", pybind11::overload_cast<const std::string, const int>(&ProductGraphGenerators::solveWithCostTimeRatio));
    smm.def("solve_with_cost_time_ratio", pybind11::overload_cast<const std::string>(&ProductGraphGenerators::solveWithCostTimeRatio));

    smm.def("solve_with_minimum_cost", pybind11::overload_cast<const std::string, const int>(&ProductGraphGenerators::solveWithDijkstra));
    smm.def("solve_with_minimum_cost", pybind11::overload_cast<const std::string>(&ProductGraphGenerators::solveWithDijkstra));

    smm.def("set_max_depth_cost_time_ratio", &ProductGraphGenerators::setMaxDepth);
    smm.def("set_cost_time_ratio_mode", &ProductGraphGenerators::setCostTimeRatioMode);

}
