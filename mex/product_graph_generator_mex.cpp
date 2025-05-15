#include <mex.h>

#include <igl/matlab/parse_rhs.h>
#include <iostream>

#include <product_graph_generators/product_graph_generators.hpp>

std::string costTimeIdToName(const int id) {
    std::string name = "";
    if (id == -1)
        name = "nomode";
    if (id == 0)
        name = "vanilla";
    else if (id == 1)
        name = "feature";
    else if (id == 2)
        name = "lengthnormalisation";
    else if (id == 3)
        name = "plainso3";
    else if (id == 4)
        name = "robustso3";
    else if (id == 5)
        name = "tbd";
    else
        std::cout << "[CPG-MEX] error: id not supported" << std::endl;
    return name;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) {
    // check and read mex input
    if (!((nrhs == 8) || ( nrhs == 9) || ( nrhs == 10) || ( nrhs == 12) || ( nrhs == 14) || ( nrhs == 17) ) || nlhs > 3) {
        mexErrMsgTxt("Usage: [passes_layers_once, minimum_mean_mean, minimum_mean_matching] = product_graph_generator_mex(VX, EX, VY, EY, feat_diff, use_conjugate_graph, use_regulariser, mean_cycle_algo, prune_intralayer, maxDepth, ctCostName, ctTimeName, NX, NY, rl_alpha, rl_c, rl_gamma)");
    }

    /*
     parse inputs
     */
    Eigen::MatrixXd VX;
    igl::matlab::parse_rhs_double(&prhs[0], VX);

    Eigen::MatrixXd EXd;
    igl::matlab::parse_rhs_double(&prhs[1], EXd);
    Eigen::MatrixXi EX = EXd.cast<int>();
    EX = EX.array() - 1; // matlab to c++

    Eigen::MatrixXd VY;
    igl::matlab::parse_rhs_double(&prhs[2], VY);

    Eigen::MatrixXd EYd;
    igl::matlab::parse_rhs_double(&prhs[3], EYd);
    Eigen::MatrixXi EY = EYd.cast<int>();
    EY = EY.array() - 1; // matlab to c++


    Eigen::MatrixXd NX, NY;
    if (nrhs >= 14) {
        igl::matlab::parse_rhs_double(&prhs[12], NX);
        igl::matlab::parse_rhs_double(&prhs[13], NY);
    }
    double rl_alpha, rl_c, rl_power;
    if (nrhs >= 17) {
        rl_alpha  = (double) *mxGetPr(prhs[14]);
        rl_c = (double) *mxGetPr(prhs[15]);
        rl_power = (double) *mxGetPr(prhs[16]);
    }
    int maxDepth;
    if (nrhs >= 10) {
        maxDepth = (int) *mxGetPr(prhs[9]);
    }
    bool pruneIntralayerEdges = false;
    if (nrhs >= 9) {
        pruneIntralayerEdges = (bool) *mxGetPr(prhs[8]);
        std::cout << "pruneIntralayerEdges " << pruneIntralayerEdges << std::endl;
    }
    std::string costName = "", timeName = "";
    if (nrhs >= 12) {
        const int costId = (int) *mxGetPr(prhs[10]);
        const int timeId = (int) *mxGetPr(prhs[11]);
        costName = costTimeIdToName(costId);
        timeName = costTimeIdToName(timeId);
    }

    Eigen::MatrixXd featDiff;
    igl::matlab::parse_rhs_double(&prhs[4], featDiff);

    const int use_conjugate_graph = (int) *mxGetPr(prhs[5]);
    const bool use_regulariser = (bool) *mxGetPr(prhs[6]);

    // strings are more complicated...
    const int mean_cycle_algo_id = (int) *mxGetPr(prhs[7]);
    std::string mean_cycle_algo = "hartmannorlin";
    if (mean_cycle_algo_id == 5)
        mean_cycle_algo = "lawlercpu";
    else if (mean_cycle_algo_id == 6)
        mean_cycle_algo = "lawlergpu";
    else if (mean_cycle_algo_id == 7)
        mean_cycle_algo = "dijkstracpu";
    else if (mean_cycle_algo_id == 8)
        mean_cycle_algo = "dijkstragpu";
    else if (mean_cycle_algo_id == -1)
        mean_cycle_algo = "export";

    /*
     run
     */
    ProductGraphGenerators pgen = ProductGraphGenerators(VX, EX, VY, EY, featDiff, use_conjugate_graph, use_regulariser, pruneIntralayerEdges);
    if (use_conjugate_graph) {
        pgen.setNormals(NX, NY);
        if (nrhs >= 17) {
            pgen.updateRobustLossParams(rl_alpha, rl_c, rl_power);
        }
    }
    if (nrhs >= 10) {
        pgen.setMaxDepth(maxDepth);
    }
    if (nrhs >= 12) {
        pgen.setCostTimeRatioMode(costName, timeName);
    }

    pgen.generate();

    std::tuple<bool, float, Eigen::MatrixXi> out;
    if (mean_cycle_algo_id > 6) {
        out = pgen.solveWithDijkstra(mean_cycle_algo);
    }
    else if (mean_cycle_algo_id > 4) {
        out = pgen.solveWithCostTimeRatio(mean_cycle_algo);
    }
    else {
        std::cout << "solver not supported" << std::endl;
    }
    const bool passes_layers_once = std::get<0>(out);
    const double minimum_mean_mean = std::get<1>(out);
    Eigen::MatrixXi minimum_mean_matching = std::get<2>(out);


    /*
     c++ to matlab
     */
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double* passes_layers_once_ptr = mxGetPr(plhs[0]);
    passes_layers_once_ptr[0] = passes_layers_once;

    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    double* minimum_mean_mean_ptr = mxGetPr(plhs[1]);
    minimum_mean_mean_ptr[0] = minimum_mean_mean;

    minimum_mean_matching = minimum_mean_matching.array() + 1; // c++ to matlab
    plhs[2] = mxCreateDoubleMatrix(minimum_mean_matching.rows(), minimum_mean_matching.cols(),  mxREAL);
    std::copy(minimum_mean_matching.data(), minimum_mean_matching.data() + minimum_mean_matching.size(), mxGetPr(plhs[2]));
}
