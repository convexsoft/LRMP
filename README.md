# LRMP

This repository provides code for Laplacian-Regularized Minimization Problems (LRMPs), with a focus on the Laplacian-Regularized Nonnegative Least-Squares (LR-NNLS) problem. It implements a Difference-of-Convex Regularizer (DCR) graph learning framework that approximates the spectral action of the Laplacian pseudoinverse without explicitly computing matrix inverses. Numerical experiments demonstrate improved accuracy and robust performance across different graph topologies.

---

## Installation

The following software and libraries are required:

- Python 3.11
- PyTorch 2.5.1
- MATLAB R2018a
- CVX for MATLAB
- CVXPY for Python
- DCCP


## Repository Structure

This repository contains the following scripts:

- `example_lrnnls_spectral_challenges.py`  
  Demonstrates the spectral challenges of the Laplacian-regularized nonnegative least-squares problem.

- `cvxpy_verify_general_solution_of_lrnnls.py`  
- `cvxpy_verify_general_solution_of_lrnnls.m`  
  Python and MATLAB codes for verifying the correctness of the general solution of the Laplacian-regularized nonnegative least-squares problem.

- `example_regu_mle_shrinkage_cccp_vs_mmcvxpy.py`  
  Implements the core components of the DCR algorithm and compares them with a CVXPY-based reference method.


## Usage

### Running Python scripts

Python scripts, such as `example_lrnnls_spectral_challenges.py`, can be executed in two ways.

#### 1. Direct execution

Open the file in a Python IDE, such as PyCharm, and run it directly.

#### 2. Command-line execution

From a terminal, navigate to the script directory and run:

```bash
python example_lrnnls_spectral_challenges.py
```

### Running MATLAB scripts

MATLAB scripts, such as `cvxpy_verify_general_solution_of_lrnnls.m`, can be executed as follows:

#### 1. Open MATLAB and navigate to the script directory

#### 2. Initialize CVX
```matlab
cvx_setup
```

#### 3. Run the script
```matlab
cvxpy_verify_general_solution_of_lrnnls
```
