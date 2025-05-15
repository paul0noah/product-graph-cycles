# Higher-Order Ratio Cycles for Fast and Globally Optimal Shape Matching

Official repository for the CVPR paper: Higher-Order Ratio Cycles for Fast and Globally Optimal Shape Matching by Paul Roetzer, Viktoria Ehm, Daniel Cremers, Zorah Lähner, Florian Bernard.

For more information please visit our [project page](https://paulroetzer.github.io/publications/2025-01-30-higher-order-ratio-cycles.html).

## ⚙️ Installation

This repo comes with c++/cuda code, python code and a matlab mex wrapper 💡

Note: Code only tested on unix systems. Expect issues with Windows 🪟.

### 🐍 Python

Simply download the repo and `python setup.py install` (you need working c++ compiler + cmake installed on your machine and working cuda compiler for cuda support). Note: `setup.py` adds cuda automatically if it finds `nvcc` compiler.

### 🔺 Matlab
💡 In the below command you can set `-DWITH_CUDA=true` to compile with cuda support
```bash
mkdir build
cd build
cmake .. -DBUILD_MEX_FILE=True -DWITH_CUDA=false -DMatlab_MEX_EXTENSION="mexmaca64" -DMatlab_ROOT_DIR=/Applications/MATLAB_R2023b.app/
make -j 8 product_graph_generator_mex
```
After building you need to copy the file `build/mex/product_graph_generator_mex.*` to the project that you want to run the mex files in!

For building, you need to adjust the variables `Matlab_MEX_EXTENSION` and `Matlab_ROOT_DIR` depending on your system!
- `Matlab_MEX_EXTENSION` is either `mexmaca64` (arm mac), `mexa64` (linux), `mexmaci64` (intel mac), `mexw64` (windows),
- `Matlab_ROOT_DIR` should point to your matlab installation

### 🚀 Cpp

```bash
mkdir build
cd build
cmake ..
make -j
```

An example script is provided in `main.cpp`.

## ✨ Usage

### 🐍 Python
All matrices are assumed to be numpy type.
```python
### Setup
import cyclic_product_graphs as cpg
import numpy as np
VX = ... # vertices of shape X
EX = ... # directed edges of shape X (e.g. of a triangle mesh), directed means that you need to provide BOTH orientations of an edge, e.g. [x1, x2] AND [x2, x1]
VY = ... # vertices of shape Y
EY = ... # directed edges of closed curve of shape Y assumed to be ordered (i.e. [[0 1], [1 4], [4, 123], [123, 17], ...])
feat_diff = ... # |VX|x|VY| matrix containing feature difference between every vertex of shape X and Y

use_conjugate_graph = False
use_regularising_cost_term = False
pg_generator = cpg.product_graph_generator(VX, EX, VY, EY, feat_diff, use_conjugate_graph, use_regularising_cost_term)
# you can also call pg_generator = cpg.product_graph_generator(VX, EX, VY, EY, feat_diff, use_conjugate_graph, use_regularising_cost_term, prune_intralayer_edges)
if use_conjugate_graph:
    NX = ... # normals of shape X, |VX|x3 
    NY = ... # normals of shape Y, |VY|x3 
    pg_generator.set_normals(NX, NY)

### Get optimal cycles
max_depth = 2 # this is essentially the distortion bound k

# solve with cost time ratio
cost_time_ratio_solver = "lawlercpu" # also possible "lawlergpu"
cost_mode = "vanilla"
time_mode = "nomode" # => mean problem
pg_generator.set_cost_time_ratio_mode(cost_mode, time_mode);
passes_layers_just_once, final_objective, matching = pg_generator.solve_with_cost_time_ratio(cost_time_ratio_solver, max_depth)  

# solve with minimum cost cycle
minimum_cost_solver = "dijkstracpu" # also possible "dijkstragpu"
passes_layers_just_once, final_objective, matching = pg_generator.solve_with_minimum_cost(minimum_cost_solver, max_depth)  

if use_conjugate_graph
    vertex_ids_Y = minimum_mean_matching[:, 0]
    vertex_ids_X = minimum_mean_matching[:, 3]
else:
    vertex_ids_Y = minimum_mean_matching[:, 0]
    vertex_ids_X = minimum_mean_matching[:, 2]
matching_x_to_y = np.column_stack((vertex_ids_X, vertex_ids_Y))
```

### 🔺 Matlab
```matlab
VX = ...; % vertices of shape X
EX = ...; % directed edges of shape X (e.g. of a triangle mesh)
VY = ...; % vertices of shape Y
EY = ...; % directed edges of closed curve of shape Y assumed to be ordered (i.e. [[0 1], [1 4], [4, 123], [123, 17], ...])
feat_diff = ...; % |VX|x|VY| matrix containing feature difference between every vertex of shape X and Y

prune_intralayer = false;
use_conjugate_graph = false;
use_regularising_cost_term = false;
solver = 5; % could also be "lawlercpu" => 5,  "lawlergpu" => 6, "dijkstracpu" => 7,  "dijkstragpu" => 8
max_depth = 2; % essentially the distortion bound k in our paper
cost_id = 0; % "nomode" => -1, "vanilla" => 0, "feature" => 1, "lengthnormalisation" => 2, "plainso3" => 3, "robustso3" => 4, "penalisedegenerate" => 5
time_id = 0; % not all id combinations between cost and time are implemented!


if use_conjugate_graph
     NX = ...; # normals of shape X, expected shape |VX|x3 
     NY = ...; # normals of shape Y, expected shape |VY|x3 
    [passes_layers_just_once, final_objective, matching] = product_graph_generator_mex(VX, EX, VY, EY, feat_diff,...
                                                                     use_conjugate_graph, use_regularising_cost_term, solver,...
                                                                     prune_intralayer, max_depth, cost_id, time_id, NX, NY);
    vertex_ids_Y = matching(:, 1);
    vertex_ids_X = matching(:, 4);
else
    [passes_layers_just_once, final_objective, matching] = product_graph_generator_mex(VX, EX, VY, EY, feat_diff,...
                                                                     use_conjugate_graph, use_regularising_cost_term, ...
                                                                     solver, prune_intralayer, max_depth, cost_id);
    vertex_ids_Y = matching(:, 1);
    vertex_ids_X = matching(:, 3);
end

matching_x_to_y = [vertex_ids_X, vertex_ids_Y]
```


### 🚀 Cpp
see file `mex/product_graph_generator_mex.cpp`

## ⚠️ Troubleshooting
- error: `CMAKE_CUDA_ARCHITECTURES not defined`, add to your `cmake ..` call  -DCMAKE_CUDA_ARCHITECTURES=XY` where XY is your architecture
- error: `This program was not compiled for SM XY `, add to your `cmake ..` call  -DCMAKE_CUDA_ARCHITECTURES=XY`


# 🎓 Attribution
If you use our code (or variants of it) please cite

```bibtex
@inproceedings{roetzer2025higherorder,
    author    = {Paul Roetzer and Viktoria Ehm and Daniel Cremers and and Zorah L\"ahner and Florian Bernard},
    title     = {Higher-Order Ratio Cycles for Fast and Globally Optimal Shape Matching},
    booktitle = {CVPR},
    year      = 2025
}
```
