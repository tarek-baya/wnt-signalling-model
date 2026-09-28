\# Wnt Signalling Model



Numerical simulation code for a reduced nonlocal surface PDE–ODE model of Wnt signalling, implemented with DUNE-FEM, PETSc, and MPI.



This repository contains code developed as part of my PhD in Mathematics at the University of Warwick on the mathematical modelling, analysis, and numerical simulation of Wnt signalling and Lipid Rafts on cell membranes.



The underlying model describes the coupled evolution of membrane-bound receptor populations and membrane lipid organisation. The membrane dynamics are represented by nonlinear surface partial differential equations, while the fast-diffusion reduction of the intracellular and extracellular bulk variables leads to a nonlocal reduced formulation involving an extracellular Wnt ODE and an intracellular receptor algebraic constraint.

The numerical implementation includes nonlinear cross-diffusion, degenerate mobilities, receptor reactions, lipid phase separation, adaptive time stepping, adaptive mesh refinement, mass-conserving admissibility corrections, and parallel MPI-based solution of the coupled surface system.

The code is intended primarily as research software accompanying the mathematical and numerical analysis of the model.



\## Repository structure



\- `wntMain.py` — main simulation driver

\- `config.py` — physical and numerical parameters

\- `weak\_forms.py` — finite-element weak formulations and operators

\- `solverMPI.py` — Newton–Schur solver

\- `mesh\_setup.py` — mesh setup, initial conditions, adaptive mesh refinement, and checkpoint loading

\- `math\_helpers.py` — model functions and numerical helper routines

\- `ode\_solver.py` — reduced bulk ODE and algebraic coupling

\- `io\_utils.py` — checkpointing, output, and plotting

\- `gmsh2dgf.py` — mesh conversion utility

\- `meshes/` — mesh files and mesh-generation utilities



\## Requirements



The code uses:



\- DUNE-FEM

\- DUNE-ALUGrid

\- PETSc / petsc4py

\- MPI

\- NumPy

\- SciPy

\- Matplotlib

\- meshio



\## Running the code



The main script is



```bash

python wntMain.py

```



For parallel execution, for example with four MPI ranks,



```bash

mpirun -n 4 python wntMain.py

```



\## Setting up an experiment



Experiment-specific settings should be added near the top of `wntMain.py` under



```python

\# =============================================================================

\# EXPERIMENT-SPECIFIC SETTINGS

\# =============================================================================

```



For example,



```python

cfg.Experiment = "experiment\_name"

cfg.SubExperiment = "variant\_name"

cfg.DescriptionExperiment = "Short description of the experiment."

cfg.AimExperiment = "Scientific aim of the experiment."



\# Change any experiment-specific model or numerical parameters here

cfg.InitialCondition = "default"

cfg.W\_b\_init = 1.0



cfg.update()

```



`cfg.Experiment`, `cfg.SubExperiment`, `cfg.DescriptionExperiment`, and `cfg.AimExperiment` are used to identify and document each numerical experiment.



Any parameters changed specifically for an experiment should be overridden in this section rather than by changing the default values in `config.py`.



Keep



```python

cfg.update()

```



after the experiment-specific parameter changes so that all derived quantities are recomputed consistently.



\## Initial conditions



The initial condition is selected through



```python

cfg.InitialCondition

```



The available generic initial conditions are defined in `mesh\_setup.py`.



\## Restarting simulations



The run mode is selected through



```python

cfg.load\_mode

```



with



```python

cfg.load\_mode = "fresh"

cfg.load\_mode = "resume"

cfg.load\_mode = "ic\_only"

```



for a new run, continuation from a checkpoint, or using a checkpoint as a new initial condition, respectively.



\## Output



The code can generate VTK files, plots of the reduced bulk variables, JSON metadata, experiment records, and restart checkpoints.



Generated simulation output is excluded from the repository through `.gitignore`.

