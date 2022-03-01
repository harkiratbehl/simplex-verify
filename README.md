# Simplex Verify

This repository contains code for Simplex Verify.

## Neural Network bounds
The repository provides code for algorithms to compute output bounds for ReLU-based neural networks (and, 
more generally, piecewise-linear networks, which can be transformed into equivalent ReLUs). The main files are in the plnn/simplex_solver folder.
- `Baseline_LinearizedNetwork` in `plnn/simplex_solver/baseline_gurobi_linear_approximation.py` represents the [PLANET](https://github.com/progirep/planet) relaxation of the network in Gurobi 
and uses the commercial solver to compute the model's output bounds.
- `DP_LinearizedNetwork` in `plnn/simplex_solver/disjunctive_gurobi.py` represents the proposed **Simplex Relaxation** relaxation of the network in Gurobi 
and uses the commercial solver to compute the model's output bounds.
- `Baseline_SimplexLP` in `plnn/simplex_solver/baseline_solver.py` implements the Opt-Lirpa Planet baseline in PyTorch, based on the Planet relaxation.
- `SimplexLP` in `plnn/simplex_solver/solver.py` implements the **Simplex Verify** algorithm presented in the paper, a
fast lirpa style algorithm for our proposed relaxation.
- `plnn/simplex_solver/baseline_cut_anderson_optimization.py` is the file corresponding to the [Active Sets](https://github.com/oval-group/scaling-the-convex-barrier)


These classes offer two main interfaces (see, for instance `tools/bounding_tools/simplex_cifar_bound_comparison.py` for detailed
usage, including algorithm parametrization):
- Given some pre-computed intermediate bounds, compute the bounds on the neural network output:
call `build_model_using_intermediate_net`, then `compute_lower_bound`.
- Compute bounds for activations of all network layers, one after the other (each layer's computation will use the
bounds computed for the previous one): `define_linear_approximation`.


## Repository structure
* `./plnn/` contains the code for the various algorithms.
* `./tools/` contains code to interface the bounds computation classes.
* `./scripts/` is a set of python scripts that, via `./tools`, run the paper's experiments.
* `./data/advertorch/` contains the trained neural network employed for both the complete and incomplete verification experiments.
  
## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [The Gurobi solver](http://www.gurobi.com/) to solve the LP arising from the
Network linear approximation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 

  
### Installation
We assume the user's Python environment is based on Anaconda.

```bash
cd code_simplex_verify

#Create a conda environment
conda create -n simplex-verify python=3.6
conda activate simplex-verify

# Install gurobipy 
conda config --add channels http://conda.anaconda.org/gurobi
pip install .
#might need
#conda install gurobi

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
# modify the cuda version to your version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 

# Install the code of this repository
python setup.py install

# Install mmbt dependencies (required for multi-modal experiments)
pip install torch torchvision sklearn pytorch-pretrained-bert numpy tqdm matplotlib

# Install advertorch dependencies (required for adversarial training and pgd bounds)
cd advertorch
python setup.py install
cd ..
```

### Execution
Finally, all l<sub>1</sub> robustness verification experiments can be replicated by running
```bash
python scripts/run_simplex_incomplete.py
```
