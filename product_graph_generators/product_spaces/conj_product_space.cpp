//
//  combinations.cpp
//  helper
//
//  Created by Paul Rötzer on 28.04.21.
//

#include "conj_product_space.hpp"
#include "helper/utils.hpp"



/* function createDataStructures
   -> build a map from all vertices to the indices of their outgoing edges
 */
std::tuple<int, int, std::vector<tsl::robin_set<int>>> ConjProductSpace::createDataStructures() {
    const int numVerticesX = EX.maxCoeff() + 1;
    std::vector<tsl::robin_set<int>> vertex2OutgoingEdgeMap;
    vertex2OutgoingEdgeMap.reserve(numVerticesX);
    for (int i = 0; i < numVerticesX; i++) {
        vertex2OutgoingEdgeMap.push_back(tsl::robin_set<int>());
    }

    Eigen::MatrixXi perVertexOutgoingEdgeCounter(numVerticesX, 1); perVertexOutgoingEdgeCounter.setZero();
    for (int edgeId = 0; edgeId < EX.rows(); edgeId++) {
        const int srcVertexId = EX(edgeId, 0);
        vertex2OutgoingEdgeMap.at(srcVertexId).insert(edgeId);
        perVertexOutgoingEdgeCounter(srcVertexId) += 1;
    }

    int numOutgoingEdges = 0;
    for (int edgeId = 0; edgeId < EX.rows(); edgeId++) {
        numOutgoingEdges += perVertexOutgoingEdgeCounter(EX(edgeId, 1)) - 1;
    }
    //perVertexOutgoingEdgeCounter = perVertexOutgoingEdgeCounter.array() - 1;
    //const int numOutgoingEdges = (perVertexOutgoingEdgeCounter.array() * perVertexOutgoingEdgeCounter.array()).sum();

    return std::make_tuple(numVerticesX, numOutgoingEdges, vertex2OutgoingEdgeMap);
}

/*function computeCombinations(...)
 Consider the edge matrix of shape X (both orientations bc triangle mesh) and the one of shape Y (single edge orientation bc 3D contour)
 -> EX is the triangle mesh
 -> EY is the contour

 -> Productspace is esentially the edges in the product graph

 returns:
    productspace = [[src2d mid2d trgt2d src3d mid3d trgt3d]
                    [src2d mid2d trgt2d src3d mid3d trgt3d]...]

 NOTES (definitely no documentation!):
  Conjugate product space
  - nodes are edges now
  - each node consists of 2 2D vertices and 2 3d vertices == single product edge
  - each edge contains therefore 8 elements


  num nodes per layer
  - #3 * numEdges3d
     1) #numEdges3d: non-degenerate (connect to next layer)
     2) #numEdges3d: degenerate 2d (connect to same layer)
     3) #numEdges3d: degenerate 3d (connect to next layer)
  num edges per layer
  - assume each vertex connected to c edges(or vertices) (in fact this has to be counted for every vertex individually then)
  - 2 * numEdges3d * 5 * (c-1)^2 + 2 * numEdges3d
  - the minus 1 comes from the fact that we do not connect an edge to its opposite direction on 3d mesh, see conj. 2d3d paper
  
*/
void ConjProductSpace::computeCombinations() {
    const std::tuple<int, int, std::vector<tsl::robin_set<int>>> dataStructures = createDataStructures();
    const int numVerticesX = std::get<0>(dataStructures);
    const int numOutgoingEdges = std::get<1>(dataStructures);
    const std::vector<tsl::robin_set<int>> vertex2OutgoingEdgeMap = std::get<2>(dataStructures);
    const int numEdgesX = (int)EX.rows();
    const int numNodesPerLayer = 3 * numEdgesX;
    const int indexOffsetDeg2d = numEdgesX;
    const int indexOffsetDeg3d = pruneIntraLayerEdges ? indexOffsetDeg2d : 2 * numEdgesX; // deg2d dont exist 

    const long numEdgesY = EY.rows();
    const long numIntraLayerEdges = 2 * numOutgoingEdges;
    const long numInterLayerEdges = 3 * numOutgoingEdges + 2 * numEdgesX;
    const long numElementsPSpace = 2 * numEdgesY * (numIntraLayerEdges + numInterLayerEdges);
    productspace = Eigen::MatrixXi(numElementsPSpace, 6);
    productspace.setZero();
    SRC_IDs  = Eigen::MatrixXi(numElementsPSpace, 1);
    TRGT_IDs = Eigen::MatrixXi(numElementsPSpace, 1);


    long numadded = 0;
    int currentstartidx = 0;
    for (int i = 0; i < numEdgesY; i++) {

        // avoid self connections
        const int nextYidx = (i+1) % numEdgesY;
        const int currentLayerStartIdx = i * (3 * numEdgesX);
        const int nextLayerStartIdx = nextYidx * (3 * numEdgesX);

        const int src2d = EY(i, 0);
        const int mid2d = EY(i, 1); //==EY(nextYidx, 0)
        const int trgt2d = EY(nextYidx, 1);

        // intralayer (deg2d -> deg2d, deg2d -> non-deg) we avoid: deg2d -> deg3d
        int noe = 0;
        if (!pruneIntraLayerEdges) {
            for (int j = 0; j < numEdgesX; j++) {
                const int src3d = EX(j, 0);
                const int mid3d = EX(j, 1); // == EX(it, 0)
                for (const auto& it : vertex2OutgoingEdgeMap.at(mid3d)) {
                    const int trgt3d = EX(it, 1);
                    if (trgt3d == src3d) continue; // avoid backpaths
                    noe++;
                    // deg2d -> deg2d
                    productspace.row(numadded) << src2d, src2d, src2d, src3d, mid3d, trgt3d;
                    SRC_IDs(numadded) = currentLayerStartIdx + indexOffsetDeg2d + j;
                    TRGT_IDs(numadded) = currentLayerStartIdx + indexOffsetDeg2d + it;
                    numadded++;

                    // deg2d -> non-deg,
                    productspace.row(numadded) << src2d, src2d, mid2d, src3d, mid3d, trgt3d;
                    SRC_IDs(numadded) = currentLayerStartIdx + indexOffsetDeg2d + j;
                    TRGT_IDs(numadded) = currentLayerStartIdx + it;
                    numadded++;
                }
            }
        }

        // interlayer (non-deg -> non-deg, non-deg->deg2d, non-deg->deg3d, deg3d -> non-deg, deg3d->deg3d) we avoid deg3d->deg2d
        for (int j = 0; j < numEdgesX; j++) {
            const int src3d = EX(j, 0);
            const int mid3d = EX(j, 1); // == EX(it, 0)
            for (const auto& it : vertex2OutgoingEdgeMap.at(mid3d)) {
                const int trgt3d = EX(it, 1);
                if (trgt3d == src3d) continue; // avoid backpaths
                //non-deg -> non-deg
                productspace.row(numadded) << src2d, mid2d, trgt2d, src3d, mid3d, trgt3d;
                SRC_IDs(numadded) = currentLayerStartIdx + j;
                TRGT_IDs(numadded) = nextLayerStartIdx + it;
                numadded++;

                // non-deg->deg2d
                if (!pruneIntraLayerEdges) {
                    productspace.row(numadded) << src2d, mid2d, mid2d, src3d, mid3d, trgt3d;
                    SRC_IDs(numadded) = currentLayerStartIdx + j;
                    TRGT_IDs(numadded) = nextLayerStartIdx + indexOffsetDeg2d + it;
                    numadded++;
                }

                // deg3d -> non-deg
                productspace.row(numadded) << src2d, mid2d, trgt2d, mid3d, mid3d, trgt3d;
                SRC_IDs(numadded) = currentLayerStartIdx + indexOffsetDeg3d + j;
                TRGT_IDs(numadded) = nextLayerStartIdx + it;
                numadded++;
            }
            // non-deg->deg3d,
            productspace.row(numadded) << src2d, mid2d, trgt2d, src3d, mid3d, mid3d;
            SRC_IDs(numadded) = currentLayerStartIdx + j;
            TRGT_IDs(numadded) = nextLayerStartIdx + indexOffsetDeg3d + j;
            numadded++;

            // deg3d->deg3d
            productspace.row(numadded) << src2d, mid2d, trgt2d, mid3d, mid3d, mid3d;
            SRC_IDs(numadded) = currentLayerStartIdx + indexOffsetDeg3d + j;
            TRGT_IDs(numadded) = nextLayerStartIdx + indexOffsetDeg3d + j;
            numadded++;
        }



    }
    if (DEBUG_CONJ_COMBINATIONS) std::cout << prefix << "Detected " << numContours << " closed contours" << std::endl;
    // the resize is very likely not necessary
    productspace.conservativeResize(numadded, 6);
    SRC_IDs.conservativeResize(numadded, 1);
    TRGT_IDs.conservativeResize(numadded, 1);

    branchGraph = std::vector<tsl::robin_set<long>>();
    std::vector<std::vector<int>> branchGraph2;
    branchGraph.reserve(numNodesPerLayer);
    for (int j = 0; j < numNodesPerLayer; j++) {
        branchGraph.push_back(tsl::robin_set<long>());
    }
    for (int j = 0; j < numEdgesX; j++) {
        branchGraph.at(j                ).insert(j +     numEdgesX);
        if (!pruneIntraLayerEdges) {
            branchGraph.at(j +     numEdgesX).insert(j + 2 * numEdgesX);
            branchGraph.at(j + 2 * numEdgesX).insert(j);
        }
        else {
            branchGraph.at(j +     numEdgesX).insert(j);
        }

        const int src3d = EX(j, 0); // == EX(it, 0)
        for (const auto& it : vertex2OutgoingEdgeMap.at(src3d)) {
            const long edgeIdx = it;
            branchGraph.at(j                ).insert(edgeIdx);
            //branchGraph.at(j +     numEdgesX).insert(edgeIdx);
            //branchGraph.at(j + 2 * numEdgesX).insert(edgeIdx);

            //branchGraph.at(j                ).insert(edgeIdx + numEdgesX);
            branchGraph.at(j +     numEdgesX).insert(edgeIdx + numEdgesX);
            //branchGraph.at(j + 2 * numEdgesX).insert(edgeIdx + numEdgesX);

            //branchGraph.at(j                ).insert(edgeIdx + 2 * numEdgesX);
            //branchGraph.at(j +     numEdgesX).insert(edgeIdx + 2 * numEdgesX);
            if (!pruneIntraLayerEdges) branchGraph.at(j + 2 * numEdgesX).insert(edgeIdx + 2 * numEdgesX);
        }


        const int trgt3d = EX(j, 0); // == EX(it, 0)
        for (const auto& it : vertex2OutgoingEdgeMap.at(trgt3d)) {
            const long edgeIdx = it;
            branchGraph.at(j                ).insert(edgeIdx);
            //branchGraph.at(j +     numEdgesX).insert(edgeIdx);
            //branchGraph.at(j + 2 * numEdgesX).insert(edgeIdx);

            //branchGraph.at(j                ).insert(edgeIdx + numEdgesX);
            branchGraph.at(j +     numEdgesX).insert(edgeIdx + numEdgesX);
            //branchGraph.at(j + 2 * numEdgesX).insert(edgeIdx + numEdgesX);

            //branchGraph.at(j                ).insert(edgeIdx + 2 * numEdgesX);
            //branchGraph.at(j +     numEdgesX).insert(edgeIdx + 2 * numEdgesX);
            if (!pruneIntraLayerEdges) branchGraph.at(j + 2 * numEdgesX).insert(edgeIdx + 2 * numEdgesX);
        }
    }

    combosComputed = true;
}


ConjProductSpace::ConjProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY, bool pruneIntralayer) : EX(EX), EY(EY) {
    combosComputed = false;
    numContours = 1;
    prefix = "[COMBOSCONJ] ";
    pruneIntraLayerEdges = pruneIntralayer;
}


ConjProductSpace::ConjProductSpace(Eigen::MatrixXi& EX, Eigen::MatrixXi& EY) : ConjProductSpace(EX, EY, false) {
}


Eigen::MatrixXi ConjProductSpace::getConjProductSpace() {
    if (!combosComputed) {
        computeCombinations();
    }
    return productspace;
}

Eigen::MatrixXi ConjProductSpace::getPiEy() {
    std::cout << prefix << "error piEY is not implemented! returning empty matrix" << std::endl;
    return Eigen::MatrixXi(1, 1);
    if (!combosComputed) {
        computeCombinations();
    }
    return piEY;
}

Eigen::MatrixXi ConjProductSpace::getSRCIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return SRC_IDs;
}

Eigen::MatrixXi ConjProductSpace::getTRGTIds() {
    if (!combosComputed) {
        computeCombinations();
    }
    return TRGT_IDs;
}

int ConjProductSpace::getNumContours() const {
    return numContours;
}


std::vector<tsl::robin_set<long>> ConjProductSpace::getBranchGraph() {
    return branchGraph;
}
