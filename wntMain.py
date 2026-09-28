# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import sys
import numpy as np
from functools import partial
from petsc4py import PETSc   # type: ignore
from dune.common import comm   #type:ignore
from dune.fem import integrate, gridAdapt, GridMarker
from dune.fem import loadBalance as femLoadBalance
from dune.fem.function import partitionFunction
from ufl import *
from dune.ufl import *
from config import cfg, setup_output, setup_checkpoint, setup_amr, setup_adaptive_dt
from math_helpers import _ghost_update, build_lumped_mass_weights, mass_conserving_bound_correction, mass_conserving_simplex_correction, relative_L2_change
from mesh_setup import *
from weak_forms import *
from io_utils import save_checkpoint, save_experiment_record, plot_timeseries, save_results_json, load_results_json, picard_H1_error, picard_max_iterations
from ode_solver import solve_ode, update_pde_coefficients
from dune.fem import parameter
parameter.append({"fem.loadbalancing.step": 1000000})

# =============================================================================
# Only rank 0 prints output 
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None
# =============================================================================
# EXPERIMENT-SPECIFIC SETTINGS
# =============================================================================
# ADD THE NAME OF THE EXPERIMENT, SUB-EXPERIMENT, DESCRIPTION, AND AIM HERE
cfg.Experiment = "experiment_name"
cfg.SubExperiment = "variant_name"
cfg.DescriptionExperiment = "Describe the experiment."
cfg.AimExperiment = "State the aim of the experiment."

# CHANGE ANY EXPERIMENT-SPECIFIC MODEL OR NUMERICAL PARAMETERS HERE
#cfg.InitialCondition = "default"
#cfg.W_b_init = 1.0
#cfg.solver_verbose = False

# Keep cfg.update() after any parameter changes above so all derived quantities are recomputed
cfg.update()
# ==============================================================================
comm.barrier() # synchronisation checkpoint: all ranks wait here until every rank has reached this point
# ==============================================================================
# Forcing the warmup run (set up in .sh file) to exit so the rest of the code is run in parallel  
if os.environ.get("DUNE_WARMUP_ONLY") == "1":
    if comm.rank == 0:
        print(f"Experiment: {cfg.Experiment}")
        print(f"SubExperiment: {cfg.SubExperiment}")
        print("DUNE_WARMUP_ONLY=1: cache warm-up complete, exiting.")
    sys.exit(0)
# =============================================================================
# 2. OUTPUT DIRECTORIES: Create output subfolders inside RUNDIR
# =============================================================================
run_tag = f"{cfg.Experiment}_{cfg.SubExperiment}" if cfg.SubExperiment else cfg.Experiment
script_dir = os.path.dirname(os.path.abspath(__file__)) # Where is my python script?
pyName = os.path.splitext(os.path.basename(__file__))[0] # The name of the python script
working_dir = os.getcwd() # Directory in which the SLURM is running = RUNDIR
job_id = os.environ.get("SLURM_JOB_ID", "local") # Identify the job ID assigned by SLURM (default local)
job_name = os.environ.get("SLURM_JOB_NAME", f"{pyName}") # Identify the job name assigned by SLURM (default pyName)
vtk_folder = os.path.join(working_dir, f"vtkOutput_{job_id}_{job_name}_{run_tag}") # Define folder for vtk output
plots_folder = os.path.join(working_dir, f"plots_{job_id}_{job_name}_{run_tag}") # Define folder for plotting output

# CHECKPOINT PATHS: used for pickling
checkpoint_dir = os.path.join(script_dir, "checkpoints") # Define folder for these checkpoints
checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{pyName}_{run_tag}.dbf") # File that will conatin the checkpoints

# Only rank 0 creates the folders
if comm.rank == 0:
    os.makedirs(vtk_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
comm.barrier() # Other ranks wait for rank 0 

# OUTPUT PATHS
vtkOutput = os.path.join(vtk_folder, f"{pyName}_{run_tag}_")
Wb_plot = os.path.join(plots_folder, f"Wb_trajectory_{run_tag}.png")
Vb_plot = os.path.join(plots_folder, f"Vb_trajectory_{run_tag}.png")

# =============================================================================
# 3. PERMISSIONS AND FREQUENCIES
# =============================================================================
print("PERMISSIONS AND FREQUENCIES:")  # config.py
setup_output(cfg.outputFlag, vtk_folder, plots_folder, working_dir, cfg.visTime, cfg.visMaxSteps)
setup_checkpoint(cfg.checkpoint_flag, checkpoint_dir, cfg.checkpointFreq, cfg.load_mode)
setup_amr(cfg.use_adaptive_mesh, cfg.adaptFreq)
setup_adaptive_dt(cfg.use_adaptive_dt)
print(f" -- Every {cfg.detailedPrintFreq} Steps Detailed Print Statement Will Be Provided ")

# =============================================================================
# 4. GRID SETUP AND INITIAL CONDITIONS
# =============================================================================
V_b_resume = cfg.V_b_init   # Safe default. Overwritten if load_mode == "resume"
V_b_initial = cfg.V_b_init  # Initial V_b for fresh runs
M_R_initial = None          # If None, total receptor mass is computed from the initial fields

if cfg.load_mode == "resume":
    (gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, _, steps, visCounter, accepted_steps,
    prev_refineTol, prev_coarsenTol, z, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_resume, M_R, physical_parameters_changed) = load_checkpoint_resume(checkpoint_path) #  mesh_setup.py

elif cfg.load_mode == "ic_only":
    (gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, _, steps, visCounter, accepted_steps,
    prev_refineTol, prev_coarsenTol, z, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_resume, M_R, physical_parameters_changed) = load_checkpoint_ic_only(checkpoint_path) #  mesh_setup.py

elif cfg.load_mode == "fresh":
    (gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, steps, visCounter, accepted_steps,
    z, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_initial, M_R_initial, physical_parameters_changed) = setup_fresh(script_dir, cfg.mesh_type, cfg.pOrder) #  mesh_setup.py
else:
    raise ValueError(
        f"Unknown cfg.load_mode={cfg.load_mode!r}. "
        "Choose from: 'fresh', 'ic_only', 'resume'."
    )

local_area = sum(e.geometry.volume for e in gridView.elements)
totAr = comm.sum(local_area)
print(f"Total area of the grid: {totAr:.5f}") # Just to check that my grid is actually distrubted across ranks

# Domain volumes and area of boundary of outer cube (needed for ODE and algebraic constraint)
vol_in, vol_ext, outerBd3D = compute_volumes(gridView, cfg.bbox_margin) # mesh_setup.py

g_L_cnst = 2.88e5
def g_L_func(t):
    return g_L_cnst * 0.0  # Total Wnt influx through outer boundary of D

run_meta_path = os.path.join(vtk_folder, f"run_meta_{run_tag}.json")
if comm.rank == 0:
    save_results_json({"totAr": totAr, "vol_in": vol_in, "vol_ext": vol_ext, "outerBd3D": outerBd3D, "beta_bump": cfg.beta_bump,
                        "k_phs": cfg.k_phs, "k_int": cfg.k_int}, run_meta_path)
comm.barrier()

if cfg.load_mode != "resume":
    if M_R_initial is None:
        int_psiF_ic = float(integrate(psiF_h, gridView, order=10))
        int_psiA_ic = float(integrate(psiA_h, gridView, order=10))
        int_psiP_ic = float(integrate(psiP_h, gridView, order=10))
        M_R = vol_in * V_b_initial + int_psiF_ic + int_psiA_ic + int_psiP_ic
    else:
        M_R = float(M_R_initial)

    print(f" •M_R total receptors mass={M_R:.4e}")

# =============================================================================
# 5. ADAPTIVE TIMESTEPPING
# =============================================================================
dt_min, dt_max = update_dt_bounds_from_mesh(gridView) # mesh_setup.py # Adjust dt_min and dt_max
cfg.dt.value = float(np.clip(cfg.dt.value, dt_min, dt_max))  # Adjust dt to new dt_bounds
print(f" •Initial dt clamped to mesh bounds: dt={cfg.dt.value:.3e}")  

# =============================================================================
# 6. SPACE, FUNCTIONS, AND OPERATORS
# =============================================================================
# Create FEM space
space = build_space(gridView)  # weak_forms.py
# Finalise the _h fucntions after building initial conditions
psiF_h = space.interpolate(psiF_h, name="psiF_h")
psiA_h = space.interpolate(psiA_h, name="psiA_h")
psiP_h = space.interpolate(psiP_h, name="psiP_h")
muP_h = space.interpolate(muP_h, name="muP_h")
phi_h = space.interpolate(phi_h, name="phi_h")
muPhi_h = space.interpolate(muPhi_h, name="muPhi_h")

# Previous-step copies
psiF_n = psiF_h.copy(); psiA_n = psiA_h.copy()
psiP_n = psiP_h.copy(); phi_n  = phi_h.copy()

# Picard iterate holders
psiF_m1 = psiF_n.copy(); psiF_m = psiF_n.copy()
psiA_m1 = psiA_n.copy(); psiA_m = psiA_n.copy()
psiP_m1 = psiP_n.copy(); psiP_m = psiP_n.copy()

# Create test and trial functions
(psiF, zetaF, psiA, zetaA, psiP, zetaP, phi, zetaPhi, muP, omgP, muPhi, omgPhi) = build_trial_test_functions(space) # weak_forms.py

# ODE constants (for reaction terms)
W_b_const = Constant(float(z[0]), name="W_b")   # Dune Constant for W_b appearing in reaction terms (updated each step)
if cfg.load_mode == "resume":
    V_b_const = Constant(V_b_resume, name="V_b")
else:
    V_b_const = Constant(V_b_initial, name="V_b")   # Dune Constant for V_b appearing in reaction terms (updated each step)

# Building reaction terms
R_f, R_a, R_p, R_phi = build_reaction_terms(psiF, psiA, psiP, phi, psiP_h, psiF_m, psiA_m, psiP_m, phi_n, W_b_const, V_b_const) # weak_forms.py

# Defining the varaitional forms via UFL
(psi_f_LHS, psi_f_RHS, psi_a_LHS, psi_a_RHS,psiP_model, psiP_grdMu, mu_muP, mu_grdp, phi_model, phi_grdMu, 
mu_muPhi, mu_grdphi) = build_variational_forms(psiF, zetaF, psiA, zetaA, psiP, zetaP,phi, zetaPhi, muP, omgP,  # weak_forms.py
muPhi, omgPhi, psiF_n, psiA_n, psiP_n, phi_n, psiF_h, psiA_h, psiP_h, phi_h, psiF_m, psiA_m, psiP_m, R_f, R_a, R_p, R_phi)

# Building operators from and schemes UFL forms 
(psi_f_Op, psi_a_Op, psiP_model_Op, Mat_psiP_model_Op, psiP_grdMu_Op, Mat_psiP_grdMu_Op, mu_grdp_Op, Mat_mu_grdp_Op,
mu_muP_Op, Mat_mu_muP_Op, phi_model_Op, Mat_phi_model_Op, phi_grdMu_Op, Mat_phi_grdMu_Op, mu_grdphi_Op, Mat_mu_grdphi_Op,
mu_muPhi_Op, Mat_mu_muPhi_Op) = build_operators_and_schemes(psi_f_LHS, psi_f_RHS, psi_a_LHS, psi_a_RHS, psiP_model, # weak_forms.py
psiP_grdMu, mu_muP, mu_grdp, phi_model, phi_grdMu, mu_muPhi, mu_grdphi, psiP_h, phi_h)

# Lumped mass weights used by the bounded mass-conserving correction of phi
phi_lumped_mass_weights = build_lumped_mass_weights(Mat_mu_muPhi_Op) # math_helpers.py
# Lumped mass weights used by the receptor simplex correction
receptor_lumped_mass_weights = build_lumped_mass_weights(Mat_mu_muP_Op) # math_helpers.py

# Enforce admissible initial phi bounds while preserving its mass
if cfg.load_mode in ("fresh", "ic_only"):
    initial_correction_info = mass_conserving_bound_correction(phi_h, phi_lumped_mass_weights, lower=-1.0, upper=1.0) # math_helpers.py

    if initial_correction_info["corrected"]:
        print(f" •Initial phi bound correction: DOFs={initial_correction_info['corrected_dofs']} | before=[{initial_correction_info['phi_min_before']:.6e},{initial_correction_info['phi_max_before']:.6e}] | after=[{initial_correction_info['phi_min_after']:.6e},{initial_correction_info['phi_max_after']:.6e}] | lambda={initial_correction_info['lambda']:.6e} | mass error={initial_correction_info['mass_error']:.3e}")

# Initialise AMR thresholds for a new state or after changing physical parameters
if cfg.load_mode in ("fresh", "ic_only") or physical_parameters_changed:
    prev_refineTol, prev_coarsenTol = init_adapt_thresholds(phi_h, psiP_h, muP_h, muPhi_h, gridView) # mesh_setup.py

# =============================================================================
# 7. SCHUR SOLVERS
# =============================================================================
# Build the Newton-Schur solvers via obtained operators  # weak_forms.py
# Note that the solvers can also be configured via parameters also in weak_forms.py or use cfg.PAREMETER on top of this script
(schur_solver_psiP, schur_params_psiP, schur_solver_phi, schur_params_phi) = build_schur_solvers(psiP_model_Op, Mat_psiP_model_Op, psiP_grdMu_Op, Mat_psiP_grdMu_Op, 
mu_grdp_Op, Mat_mu_grdp_Op, Mat_mu_muP_Op, phi_model_Op, Mat_phi_model_Op, phi_grdMu_Op, Mat_phi_grdMu_Op, mu_grdphi_Op, Mat_mu_grdphi_Op, Mat_mu_muPhi_Op)

# =============================================================================
# 8. ADAPTIVE MESH INDICATOR
# =============================================================================
indicator, indicator_maxima = build_indicator(phi_h, psiP_h, muP_h, muPhi_h, gridView)  # mesh_setup.py

# =============================================================================
# 9. INITIAL VTK SNAPSHOT
# =============================================================================
vtk_meta = {}
vtk_meta_path = os.path.join(vtk_folder, f"vtk_meta_{run_tag}.json")
last_vtk_write_time = None
last_vtk_write_step = None

if cfg.load_mode == "resume" and os.path.exists(vtk_meta_path):
    vtk_meta = {int(k): v for k, v in load_results_json(vtk_meta_path).items()}
    if vtk_meta:
        last_vtk_write_time = vtk_meta[max(vtk_meta.keys())]["time"]
        last_vtk_write_step = vtk_meta[max(vtk_meta.keys())]["accepted_steps"]

if cfg.outputFlag and cfg.load_mode != "resume":
    gridView.writeVTK(
        vtkOutput,
        pointdata=[phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h],
        celldata=[partitionFunction(gridView)],
        number=visCounter,
    )

    if comm.rank == 0:
        vtk_meta[visCounter] = {
            "time": actTime,
            "dt": cfg.dt.value,
            "W_b": float(W_b_const.value),
            "V_b": float(V_b_const.value),
            "steps": steps,
            "accepted_steps": accepted_steps,
            "beta_bump": cfg.beta_bump,
            "k_phs": cfg.k_phs,
            "k_int": cfg.k_int,
        }
        save_results_json(vtk_meta, vtk_meta_path)

    last_vtk_write_time = actTime
    last_vtk_write_step = accepted_steps
    visCounter += 1
    comm.barrier()

if comm.rank == 0 and cfg.outputFlag:
    print("OUTPUT PATHS:")
    print(f" • W_b data: {os.path.splitext(Wb_plot)[0]}_data.json")
    print(f" • W_b plots: {os.path.splitext(Wb_plot)[0]} [.png .pdf .svg]")
    print(f" • V_b data: {os.path.splitext(Vb_plot)[0]}_data.json")
    print(f" • V_b plots: {os.path.splitext(Vb_plot)[0]} [.png .pdf .svg]")
    print(f" • VTK metadata: {vtk_meta_path}")

comm.barrier()

# =============================================================================
# 10. TIME LOOP
# =============================================================================
_last_print_grid_size = count_owned_dofs(phi_h) # mesh_setup.py # To compare grid sizes afetr change
#--------------- What has changed? ---------------#
save_experiment_record(cfg=cfg, plots_folder=plots_folder, vtk_folder=vtk_folder, checkpoint_path=checkpoint_path, g_L_func_src=g_L_func,
                    build_ic_func=build_initial_conditions, mesh_type=cfg.mesh_type, load_mode=cfg.load_mode, job_id=job_id, job_name=job_name, working_dir=working_dir)

steady_ref_phi = phi_h.copy()
steady_ref_psiF = psiF_h.copy()
steady_ref_psiA = psiA_h.copy()
steady_ref_psiP = psiP_h.copy()
steady_ref_Vb = float(V_b_const.value)
steady_ref_Wb = float(z[0])
steady_ref_time = float(actTime)
steady_success_count = 0

muP_h_saved = muP_h.copy()
muPhi_h_saved = muPhi_h.copy()
amr_fields = [
    psiF_h, psiA_h, psiP_h, muP_h, phi_h, muPhi_h,
    psiF_n, psiA_n, psiP_n, phi_n,
    psiF_m, psiA_m, psiP_m,
    psiF_m1, psiA_m1, psiP_m1,
    muP_h_saved, muPhi_h_saved,
    steady_ref_phi, steady_ref_psiF, steady_ref_psiA, steady_ref_psiP,
]

last_accepted_dt = None   # Tracks the actual dt used by the most recent ACCEPTED step, for the final VTK metadata write

while actTime < cfg.finalT:
    deltaT = min(float(cfg.dt.value), float(cfg.finalT - actTime))  # Save current dt and prevent overshooting finalT
    cfg.dt.value = deltaT  # Ensure the PDE and ODE use the same attempted time step
    steps += 1  # Increment step counter
    
    # --- Store previous step ---
    psiF_n.assign(psiF_h); psiA_n.assign(psiA_h)
    psiP_n.assign(psiP_h); phi_n.assign(phi_h)
    cfg.t.assign(actTime)
    z_n = z.copy()

    # --- Save mu fields for restoration on any rejection ---
    muP_h_saved.assign(muP_h)
    muPhi_h_saved.assign(muPhi_h)
    _ghost_update(muP_h_saved.as_petsc)
    _ghost_update(muPhi_h_saved.as_petsc)

    # --- Initialise Picard iterates ---
    psiF_m.assign(psiF_n);  psiA_m.assign(psiA_n); psiP_m.assign(psiP_n)
    psiF_m1.assign(psiF_n);  psiA_m1.assign(psiA_n); psiP_m1.assign(psiP_n)

    # -----------------------------------------------------------------------
    # STEP 1: Picard loop for psiF, psiA and psiP
    # -----------------------------------------------------------------------
    picard_converged = False
    picard_solver_failed = False
    failed_solver = None
    picard_max_it = None
    err = np.inf
    psiP_newton_max = 0

    for it in range(cfg.picardMaxIt):
        if schur_params_psiP["verbose"]:
            print(f" -•-[Picard iter {it+1}] Starting...", flush=True)

        # Reassemble psiP block matrices with current Picard iterates
        psiP_grdMu_Op.jacobian(muP_h, Mat_psiP_grdMu_Op)  # BT: depends on M_p(psiP_m, rhoR_m, phi_n)

        # Solve the psiF and psiA schemes
        info_psiF = psi_f_Op.solve(target=psiF_m1)
        if not info_psiF["converged"]:
            picard_solver_failed = True
            failed_solver = "psiF"
            break

        info_psiA = psi_a_Op.solve(target=psiA_m1)
        if not info_psiA["converged"]:
            picard_solver_failed = True
            failed_solver = "psiA"
            break

        # Solve the psiP-muP block
        if schur_params_psiP["verbose"]:
            print(f"  -•-[psiP block] Starting NEWTON solve...", flush=True)

        info_psiP = schur_solver_psiP.solve(target1=psiP_m1, target2=muP_h)
        psiP_newton_max = max(psiP_newton_max, info_psiP["iterations"])
        if not info_psiP["converged"]:
            picard_solver_failed = True
            failed_solver = "psiP"
            break

        # Compute errors
        err_abs, err, _ = picard_H1_error(psiF_m1, psiA_m1, psiP_m1, psiF_m, psiA_m, psiP_m, gridView)

        if it == 0:
            picard_max_it = picard_max_iterations(err_abs, cfg.picardTol)

            if schur_params_psiP["verbose"]:
                print(f" -•-[Picard first correction | err_abs={err_abs:.3e} | " f"m_max={picard_max_it}]", flush=True)
        
        psiF_m.assign(psiF_m1)
        _ghost_update(psiF_m.as_petsc)

        psiA_m.assign(psiA_m1)
        _ghost_update(psiA_m.as_petsc)

        psiP_m.assign(psiP_m1)
        _ghost_update(psiP_m.as_petsc)

        if err <= cfg.picardTol:
            picard_converged = True
            if schur_params_psiP["verbose"]:
                print(
                    f" -•-[Picard CONVERGED in {it+1} iters | "
                    f"err={err:.3e}] (tol={cfg.picardTol:.3e})",
                    flush=True,
                )
            break

        if it + 1 >= picard_max_it:
            break

    # Assess solver failure inside the Picard loop
    if picard_solver_failed:
        if deltaT <= dt_min * (1.0 + 1e-12):
            if failed_solver == "psiP":
                failure_details = (f"residual={info_psiP['residual']:.3e} but "
                    f"tol={cfg.newtonTol:.3e}")
            elif failed_solver == "psiF":
                failure_details = (f"iterations={info_psiF['iterations']} and "
                    f"linear_iterations={info_psiF['linear_iterations']}")
            else:
                failure_details = (f"iterations={info_psiA['iterations']} and "
                    f"linear_iterations={info_psiA['linear_iterations']}")

            raise RuntimeError(f"{failed_solver} solver failed at dt_min={dt_min:.3e}: "
                f"{failure_details}. Aborting.")

        newton_reduction_pct = (1 - cfg.dt_reduce_after_newton_reject) * 100

        if failed_solver == "psiP":
            failure_details = (f"residual={info_psiP['residual']:.3e} but "
                f"tol={cfg.newtonTol:.3e}")
        elif failed_solver == "psiF":
            failure_details = (f"iterations={info_psiF['iterations']} | "
                f"linear_iterations={info_psiF['linear_iterations']}")
        else:
            failure_details = (f"iterations={info_psiA['iterations']} | "
                                f"linear_iterations={info_psiA['linear_iterations']}")

        print(f" •{failed_solver} solver failed ({failure_details}).")
        print(f" •Reducing dt by {newton_reduction_pct:.2f}% "
            f"until it hits dt_min={dt_min:.3e}")

        cfg.dt.value = max(deltaT * cfg.dt_reduce_after_newton_reject, dt_min)
        print(f" •dt reduced! dt={cfg.dt.value:.3e} >= dt_min={dt_min:.3e}")

        steps -= 1
        muP_h.assign(muP_h_saved)
        continue

    # Assess Picard convergence
    if not picard_converged:
        if deltaT <= dt_min * (1.0 + 1e-12):
            raise RuntimeError(
                f"Picard failed at dt_min={dt_min:.3e}. "
                f"err={err:.3e} but tol={cfg.picardTol:.3e}. Aborting."
            )

        picard_reduction_pct = (1 - cfg.dt_reduce_after_picard_reject) * 100
        print(
            f" •WARNING: Picard did not converge "
            f"(err={err:.3e} but tol={cfg.picardTol:.3e}). "
            f"Reducing dt by {picard_reduction_pct:.2f}% "
            f"until it hits dt_min={dt_min:.3e}"
        )

        cfg.dt.value = max(deltaT * cfg.dt_reduce_after_picard_reject, dt_min)

        print(f" •dt reduced! dt={cfg.dt.value:.3e} >= dt_min={dt_min:.3e}")

        steps -= 1
        muP_h.assign(muP_h_saved)
        continue
    
    # Push new solutions to _h
    psiF_h.assign(psiF_m)
    psiA_h.assign(psiA_m)
    psiP_h.assign(psiP_m)

    # Enforce receptor simplex while preserving total surface receptor mass
    simplex_info = mass_conserving_simplex_correction(psiF_h, psiA_h, psiP_h, receptor_lumped_mass_weights)
    schur_solver_psiP.recompute_target2_from_constraint(psiP_h, muP_h)

    if simplex_info["corrected"]:
        print(
            f" •Receptor simplex correction: DOFs={simplex_info['corrected_dofs']} | "
            f"before: min(psi)={simplex_info['psi_min_before']:.6e}, rho_max={simplex_info['rho_max_before']:.6e} | "
            f"after: min(psi)={simplex_info['psi_min_after']:.6e}, rho_max={simplex_info['rho_max_after']:.6e} | "
            f"lambda={simplex_info['lambda']:.6e} | mass error={simplex_info['mass_error']:.3e}")
    
    # -----------------------------------------------------------------------
    # STEP 3: Newton-Schur solve for phi and muPhi
    # -----------------------------------------------------------------------
    if schur_params_phi["verbose"]:
        print(" -•-[phi block] Starting NEWTON solve.", flush=True)

    phi_grdMu_Op.jacobian(muPhi_h, Mat_phi_grdMu_Op)
    info_phi = schur_solver_phi.solve(target1=phi_h, target2=muPhi_h) # solverMPI.py

    if info_phi["converged"]:
        correction_info = mass_conserving_bound_correction(phi_h, phi_lumped_mass_weights, lower=-1.0, upper=1.0) # math_helpers.py

        if correction_info["corrected"]:
            schur_solver_phi.recompute_target2_from_constraint(phi_h, muPhi_h) # solverMPI.py
            print(f" •phi bound correction: DOFs={correction_info['corrected_dofs']} | before=[{correction_info['phi_min_before']:.6e},{correction_info['phi_max_before']:.6e}] | after=[{correction_info['phi_min_after']:.6e},{correction_info['phi_max_after']:.6e}] | lambda={correction_info['lambda']:.6e} | mass error={correction_info['mass_error']:.3e}")

    # -----------------------------------------------------------------------
    # STEP 4: Tentative ODE solve + adaptive dt check
    # -----------------------------------------------------------------------
    diverged = not info_psiP["converged"] or not info_phi["converged"]  # Did Newton solver fail to converge?

    if diverged:
        psiF_h.assign(psiF_n); psiA_h.assign(psiA_n)  # Restore old solutions
        psiP_h.assign(psiP_n); phi_h.assign(phi_n)
        muP_h.assign(muP_h_saved); muPhi_h.assign(muPhi_h_saved)  # Restore mu fields
        z = z_n.copy()
        steps -= 1  # Don't count this as an accepted step

        if deltaT <= dt_min * (1.0 + 1e-12):
            raise RuntimeError(
                f"Newton failed at dt_min={dt_min:.3e}. "
                f"psiP residual={info_psiP['residual']:.3e} and "
                f"phi residual={info_phi['residual']:.3e} but "
                f"tol={cfg.newtonTol:.3e}. Aborting."
            )

        newton_reduction_pct = (1 - cfg.dt_reduce_after_newton_reject) * 100
        print(f" •Newton failed (psiP residual={info_psiP['residual']:.3e} and phi residual={info_phi['residual']:.3e} but tol={cfg.newtonTol:.3e}).")
        print(f" •Reducing dt by {newton_reduction_pct:.2f}% until it hits dt_min={dt_min:.3e}")
        cfg.dt.value = max(deltaT * cfg.dt_reduce_after_newton_reject, dt_min)
        print(f" •dt reduced! dt={cfg.dt.value:.3e} >= dt_min={dt_min:.3e}")
        continue  # Go back to beginning of while loop and retry with new dt

    if cfg.use_adaptive_dt:
        z_tent = solve_ode(actTime, actTime + deltaT, z_n, psiF_h, psiA_h, gridView, vol_ext, g_L_func) # ode_solver.py  # Tentatively solve for W_b
        _, retry_step = adaptive_timestep(psiF_h, psiF_n, psiA_h, psiA_n, psiP_h, psiP_n, phi_h, phi_n, cfg.dt, deltaT, dt_min, dt_max) # mesh_setup.py  # Check solution changes

        if retry_step:  # If step rejected
            psiF_h.assign(psiF_n); psiA_h.assign(psiA_n)  # Restore old solutions
            psiP_h.assign(psiP_n); phi_h.assign(phi_n)
            muP_h.assign(muP_h_saved); muPhi_h.assign(muPhi_h_saved)  # Restore mu fields
            z = z_n.copy()
            steps -= 1  # Don't count this as an accepted step
            continue  # Go back to beginning of while loop and retry with new dt

        # Limit dt growth when either Schur-Newton solve is becoming difficult
        newton_max = max(psiP_newton_max, info_phi["iterations"])

        if newton_max >= 25:
            cfg.dt.value = max(min(float(cfg.dt.value), 0.6 * deltaT), dt_min)
            print(f" •Newton expensive: psiP max={psiP_newton_max} | phi={info_phi['iterations']} --> reducing next dt to {cfg.dt.value:.3e}")
        elif newton_max >= 15:
            cfg.dt.value = max(min(float(cfg.dt.value), deltaT), dt_min)
            print(f" •Newton expensive: psiP max={psiP_newton_max} | phi={info_phi['iterations']} --> holding next dt at {cfg.dt.value:.3e}")

        # If all went well
        z = z_tent  # Confirm the ODE solution
    else:  # If we are not using adaptive dt, just solve the ODE
        z = solve_ode(actTime, actTime + deltaT, z_n, psiF_h, psiA_h, gridView, vol_ext, g_L_func) # ode_solver.py

    # Update W_b and V_b PDE coefficients
    update_pde_coefficients(z, psiF_h, psiA_h, psiP_h, gridView, W_b_const, V_b_const, M_R, vol_in) # ode_solver.py
    
    # Successfully passed a step
    accepted_steps += 1
    last_accepted_dt = deltaT   # Captured now, before adaptive_timestep() may update cfg.dt.value for the NEXT proposed step
    actTime += deltaT  # Add one time step
    Wb_time_list.append(actTime)  # Time point for W_b trajectory
    Wb_vals.append(float(z[0]))  # W_b value at this time point
    Vb_time_list.append(actTime)  # Time point for V_b trajectory
    Vb_vals.append(float(V_b_const.value))  # V_b value at this time point
    
    # -----------------------------------------------------------------------
    # STEP 5: ADAPTIVE MESH REFINEMENT
    # -----------------------------------------------------------------------
    if cfg.use_adaptive_mesh and accepted_steps % cfg.adaptFreq == 0:  # Adapt frequency achieved and AMR enabled
        # Evaluate indicator at element centres (local to each rank)
        update_indicator_maxima(indicator_maxima, gridView)
        local_vals = np.array([indicator(e, e.geometry.center) for e in gridView.elements])
        local_max = float(np.max(local_vals)) if local_vals.size > 0 else 0.0
        val_max = comm.max(local_max)

        # Dynamic refine/coarsen tolerances based on current indicator magnitude
        raw_ref = cfg.raw_ref_frac * val_max
        raw_crs = cfg.raw_crs_frac * val_max

        # Limit tolerance changes to avoid sudden refinement jumps
        ref_tol = float(np.clip(raw_ref, prev_refineTol * (1 - cfg.max_tol_change), prev_refineTol * (1 + cfg.max_tol_change)))
        crs_tol = float(np.clip(raw_crs, prev_coarsenTol * (1 - cfg.max_tol_change), prev_coarsenTol * (1 + cfg.max_tol_change)))
        prev_refineTol = ref_tol
        prev_coarsenTol = crs_tol

        old_size = count_owned_dofs(phi_h) # mesh_setup.py # Size of grid before any adaptation
        mesh_adapted = False

        # Constrain adaptation level changes to the current global mesh range +-1
        local_levels = np.array([e.level for e in gridView.elements], dtype=int)
        local_min_level = int(np.min(local_levels)) if local_levels.size > 0 else cfg.maxLevel
        local_max_level = int(np.max(local_levels)) if local_levels.size > 0 else cfg.minLevel
        global_min_level = int(comm.min(local_min_level))
        global_max_level = int(comm.max(local_max_level))
        allowed_mn = max(global_min_level - 1, cfg.minLevel)
        allowed_mx = min(global_max_level + 1, cfg.maxLevel)

        if crs_tol < ref_tol:
            gridAdapt(GridMarker(indicator, refineTolerance=ref_tol, coarsenTolerance=crs_tol, minLevel=allowed_mn, maxLevel=allowed_mx, markNeighbors=True), amr_fields)  # Prolongate/restrict all fields onto new mesh without automatic load balancing

            # Check MPI partition after adaptation
            _, min_dofs, avg_dofs, max_dofs, _, imbalance, min_fraction = owned_dof_distribution(phi_h) # mesh_setup.py
            print(f" •Post-AMR partition: DOFs/rank min={min_dofs} | avg={avg_dofs:.1f} | max={max_dofs} | max/avg={imbalance:.3f} | min/avg={min_fraction:.3f}")

            # Load balance only if needed and safe
            needs_balance = imbalance > 1.5
            safe_to_balance = min_dofs >= 1000 and min_fraction >= 0.25

            if needs_balance and safe_to_balance:
                print(f" •Load balancing: imbalance={imbalance:.3f}")
                femLoadBalance(amr_fields)
            elif needs_balance:
                print(f" •Load balance skipped for safety: min DOFs/rank={min_dofs} | avg={avg_dofs:.1f} | min/avg={min_fraction:.3f} | imbalance={imbalance:.3f}")
            else:
                print(f" •Load balance not needed: imbalance={imbalance:.3f}")

            mesh_adapted = True
        else:  # Invalid or overlapping refinement/coarsening thresholds
            print(f" •AMR skipped: crs_tol={crs_tol:.3e} >= ref_tol={ref_tol:.3e}")

        if mesh_adapted:
            new_size = count_owned_dofs(phi_h) # mesh_setup.py  # Size of grid after adaptation
            
            # Updating ghost values
            _ghost_update(psiF_h.as_petsc)
            _ghost_update(psiA_h.as_petsc)
            _ghost_update(psiP_h.as_petsc)
            _ghost_update(phi_h.as_petsc)
            _ghost_update(muP_h.as_petsc)
            _ghost_update(muPhi_h.as_petsc)
            _ghost_update(psiF_n.as_petsc)
            _ghost_update(psiA_n.as_petsc)
            _ghost_update(psiP_n.as_petsc)
            _ghost_update(phi_n.as_petsc)

            # Reassemble all matrices on the new mesh DOF layout after adaptation
            # --- psiP block ---
            psiP_model_Op.jacobian(psiP_h, Mat_psiP_model_Op)
            psiP_grdMu_Op.jacobian(muP_h, Mat_psiP_grdMu_Op)
            mu_muP_Op.jacobian(muP_h, Mat_mu_muP_Op)
            mu_grdp_Op.jacobian(psiP_h, Mat_mu_grdp_Op)
            schur_solver_psiP.rebuild_after_amr() # solverMPI.py
            
            # --- phi block ---
            phi_model_Op.jacobian(phi_h, Mat_phi_model_Op)
            phi_grdMu_Op.jacobian(muPhi_h, Mat_phi_grdMu_Op)
            mu_muPhi_Op.jacobian(muPhi_h, Mat_mu_muPhi_Op)
            mu_grdphi_Op.jacobian(phi_h, Mat_mu_grdphi_Op)
            schur_solver_phi.rebuild_after_amr() # solverMPI.py

            # Rebuild lumped mass weights on the new mesh DOF layout
            phi_lumped_mass_weights.destroy()
            phi_lumped_mass_weights = build_lumped_mass_weights(Mat_mu_muPhi_Op) # math_helpers.py
            receptor_lumped_mass_weights.destroy()
            receptor_lumped_mass_weights = build_lumped_mass_weights(Mat_mu_muP_Op)

            simplex_info_amr = mass_conserving_simplex_correction(psiF_h, psiA_h, psiP_h, receptor_lumped_mass_weights)
            schur_solver_psiP.recompute_target2_from_constraint(psiP_h, muP_h)

            if simplex_info_amr["corrected"]:
                print(
                    f" •Receptor post-AMR simplex correction: DOFs={simplex_info_amr['corrected_dofs']} | "
                    f"before: min(psi)={simplex_info_amr['psi_min_before']:.6e}, rho_max={simplex_info_amr['rho_max_before']:.6e} | "
                    f"after: min(psi)={simplex_info_amr['psi_min_after']:.6e}, rho_max={simplex_info_amr['rho_max_after']:.6e} | "
                    f"mass error={simplex_info_amr['mass_error']:.3e}")
    
            # Enforce bounds and preserve phi mass after mesh transfer
            correction_info_amr = mass_conserving_bound_correction(phi_h, phi_lumped_mass_weights, lower=-1.0, upper=1.0) # math_helpers.py

            # Recompute muPhi from the final corrected phi
            schur_solver_phi.recompute_target2_from_constraint(phi_h, muPhi_h) # solverMPI.py

            if correction_info_amr["corrected"]:
                print(f" •phi post-AMR bound correction: DOFs={correction_info_amr['corrected_dofs']} | before=[{correction_info_amr['phi_min_before']:.6e},{correction_info_amr['phi_max_before']:.6e}] | after=[{correction_info_amr['phi_min_after']:.6e},{correction_info_amr['phi_max_after']:.6e}] | lambda={correction_info_amr['lambda']:.6e} | mass error={correction_info_amr['mass_error']:.3e}")
            
            # How much did my grid change?
            diff_size = new_size - old_size
            if diff_size != 0:
                print(f" •Grid changed after AMR at step={steps}: {int(old_size)} --> {int(new_size)} DOFs ({diff_size / old_size * 100:.4f}%)")
            else:
                print(f" •Grid unchanged after AMR at step={steps}: {int(new_size)} DOFs")

            if cfg.use_adaptive_dt:
                # Get new time stepping parameters after mesh change
                dt_min, dt_max = update_dt_bounds_from_mesh(gridView) # mesh_setup.py
                cfg.dt.value = float(np.clip(cfg.dt.value, dt_min, dt_max))

                # Do we need to reduce dt due to large grid change?
                if diff_size > 0 and (diff_size / old_size * 100) > cfg.max_amr_allowed:
                    amr_reduction_pct = (1 - cfg.dt_reduce_after_amr_change) * 100
                    dt_amr = deltaT * cfg.dt_reduce_after_amr_change  # AMR suggestion based on deltaT (the dt that was actually used this step)
                    dt_adaptive = cfg.dt.value  # adaptive_timestep's suggestion based on solution changes
                    cfg.dt.value = max(min(dt_adaptive, dt_amr), dt_min)  # Take the most conservative, never go below dt_min
                    print(f" •dt reduced by {amr_reduction_pct:.2f}% due to large mesh change (change={diff_size / old_size * 100:.4f}% > allowed={cfg.max_amr_allowed:.4f}%)")

                # Print final dt and bounds after all changes
                print(f" •dt after AMR: dt={cfg.dt.value:.3e} (bounds: [{dt_min:.4e}, {dt_max:.4e}])")

            PETSc.garbage_cleanup(PETSc.COMM_WORLD)
            comm.barrier()
    
    # Plotting
    if cfg.outputFlag and (last_vtk_write_time is None or last_vtk_write_step is None or actTime - last_vtk_write_time >= cfg.visTime or accepted_steps - last_vtk_write_step >= cfg.visMaxSteps):
        gridView.writeVTK(vtkOutput,
            pointdata=[psiF_h, psiA_h, psiP_h, muP_h, phi_h, muPhi_h],
            celldata=[partitionFunction(gridView)],
            number=visCounter)

        if comm.rank == 0:
            vtk_meta[visCounter] = {"time": actTime, "dt": deltaT, "W_b": float(W_b_const.value), "V_b": float(V_b_const.value), "steps": steps, "accepted_steps": accepted_steps,
                                    "beta_bump": cfg.beta_bump, "k_phs": cfg.k_phs, "k_int": cfg.k_int}
            save_results_json(vtk_meta, vtk_meta_path)

        last_vtk_write_time = actTime   # tracked on EVERY rank identically -- used later to decide whether to skip the collective final writeVTK() call
        last_vtk_write_step = accepted_steps
        visCounter += 1
        comm.barrier()    

    # periodically save a checkpoint in case simulation stops abruptly
    if cfg.checkpoint_flag and steps % cfg.checkpointFreq == 0:
        if comm.rank == 0:
            plot_timeseries(np.array(Wb_time_list), np.array(Wb_vals), ylabel=r"$W_{\mathrm{b}}$", title=r"Extracellular Wnt $W_{\mathrm{b}}(t)$", filename=Wb_plot, color="tab:orange", x_limits=(0, cfg.finalT), y_limits=None)
            plot_timeseries(np.array(Vb_time_list), np.array(Vb_vals), ylabel=r"$V_{\mathrm{b}}$", title=r"Intracellular receptors $V_{\mathrm{b}}(t)$", filename=Vb_plot, color="tab:blue", x_limits=(0, cfg.finalT), y_limits=None)

        save_checkpoint(gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, cfg.dt.value, steps, visCounter, accepted_steps,
                        prev_refineTol, prev_coarsenTol, z, M_R, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_const, checkpoint_path)
        comm.barrier()

    # Short every step printing
    print(f"Step={steps} | t={actTime:.4e} | dt={deltaT:.3e} | Picard={picard_converged} | Newton psiP={info_psiP['converged']} | Newton phi={info_phi['converged']}")

    # Long periodic detailed print
    if steps % cfg.detailedPrintFreq == 0:
        # --- Field min/max ---
        psiF_arr = psiF_h.as_petsc.getArray()[:psiF_h.as_petsc.local_size]
        psiA_arr = psiA_h.as_petsc.getArray()[:psiA_h.as_petsc.local_size]
        psiP_arr = psiP_h.as_petsc.getArray()[:psiP_h.as_petsc.local_size]
        phi_arr  = phi_h.as_petsc.getArray()[:phi_h.as_petsc.local_size]

        psiF_min = comm.min(float(psiF_arr.min())); psiF_max = comm.max(float(psiF_arr.max()))
        psiA_min = comm.min(float(psiA_arr.min())); psiA_max = comm.max(float(psiA_arr.max()))
        psiP_min = comm.min(float(psiP_arr.min())); psiP_max = comm.max(float(psiP_arr.max()))
        phi_min  = comm.min(float(phi_arr.min()));  phi_max  = comm.max(float(phi_arr.max()))

        # --- Max change per field normalised by dt ---
        psiF_arr_rate = psiF_h.as_petsc.getArray(readonly=True)[:psiF_h.as_petsc.local_size]
        psiA_arr_rate = psiA_h.as_petsc.getArray(readonly=True)[:psiA_h.as_petsc.local_size]
        psiP_arr_rate = psiP_h.as_petsc.getArray(readonly=True)[:psiP_h.as_petsc.local_size]
        phi_arr_rate = phi_h.as_petsc.getArray(readonly=True)[:phi_h.as_petsc.local_size]
        psiF_n_arr_rate = psiF_n.as_petsc.getArray(readonly=True)[:psiF_n.as_petsc.local_size]
        psiA_n_arr_rate = psiA_n.as_petsc.getArray(readonly=True)[:psiA_n.as_petsc.local_size]
        psiP_n_arr_rate = psiP_n.as_petsc.getArray(readonly=True)[:psiP_n.as_petsc.local_size]
        phi_n_arr_rate = phi_n.as_petsc.getArray(readonly=True)[:phi_n.as_petsc.local_size]

        dpsiF = comm.max(float(np.max(np.abs(psiF_arr_rate-psiF_n_arr_rate))))/deltaT
        dpsiA = comm.max(float(np.max(np.abs(psiA_arr_rate-psiA_n_arr_rate))))/deltaT
        dpsiP = comm.max(float(np.max(np.abs(psiP_arr_rate-psiP_n_arr_rate))))/deltaT
        dphi = comm.max(float(np.max(np.abs(phi_arr_rate-phi_n_arr_rate))))/deltaT
        
        # --- Grid and ODE scalars ---
        current_grid_size = count_owned_dofs(phi_h) # mesh_setup.py
        grid_change = int(current_grid_size - _last_print_grid_size)
        grid_change_str = f"+{grid_change}" if grid_change > 0 else str(grid_change)

        # --- Mass conservation check ---
        _int_psiF = float(integrate(psiF_h, gridView, order=10))
        _int_psiA = float(integrate(psiA_h, gridView, order=10))
        _int_psiP = float(integrate(psiP_h, gridView, order=10))
        M_R_check = vol_in * V_b_const.value + _int_psiF + _int_psiA + _int_psiP
        M_R_err = abs(M_R_check - M_R) / max(abs(M_R), 1e-60)

        print(f" --•psiF=[{psiF_min:.3e},{psiF_max:.3e}] | psiA=[{psiA_min:.3e},{psiA_max:.3e}] | psiP=[{psiP_min:.3e},{psiP_max:.3e}] | phi=[{phi_min:.3e},{phi_max:.3e}]")
        print(f" --•||d/dt||: psiF={dpsiF:.2e} | psiA={dpsiA:.2e} | psiP={dpsiP:.2e} | phi={dphi:.2e}")
        print(f" --•Picard iters={it+1} | Newton psiP={info_psiP['iterations']} | Lin psiP={info_psiP['linear_iters']} | Newton phi={info_phi['iterations']} | Lin phi={info_phi['linear_iters']}")
        print(f" --•W_b={z[0]:.4e} | V_b={V_b_const.value:.4e} | M_R_err={M_R_err:.2e} | grid={int(current_grid_size)} (net {grid_change_str} DOFs)")

        _last_print_grid_size = current_grid_size
        
    if cfg.use_quasi_steady_stop and actTime >= cfg.steady_min_time and actTime-steady_ref_time >= cfg.steady_check_window:
        steady_err_phi = relative_L2_change(phi_h, steady_ref_phi, gridView)
        steady_err_psiF = relative_L2_change(psiF_h, steady_ref_psiF, gridView)
        steady_err_psiA = relative_L2_change(psiA_h, steady_ref_psiA, gridView)
        steady_err_psiP = relative_L2_change(psiP_h, steady_ref_psiP, gridView)
        steady_err_Vb = abs(float(V_b_const.value)-steady_ref_Vb) / max(1.0, abs(steady_ref_Vb))
        steady_err_Wb = abs(float(z[0])-steady_ref_Wb) / max(1.0, abs(steady_ref_Wb))
        steady_error = max(steady_err_phi, steady_err_psiF, steady_err_psiA, steady_err_psiP, steady_err_Vb, steady_err_Wb)
        steady_success_count = steady_success_count+1 if steady_error < cfg.steady_tol else 0

        steady_ref_phi.assign(phi_h)
        steady_ref_psiF.assign(psiF_h)
        steady_ref_psiA.assign(psiA_h)
        steady_ref_psiP.assign(psiP_h)
        steady_ref_Vb = float(V_b_const.value)
        steady_ref_Wb = float(z[0])
        steady_ref_time = float(actTime)

        local_steady_reached = steady_success_count >= cfg.steady_required_checks
        steady_votes = int(comm.sum(1 if local_steady_reached else 0))

        if steady_votes not in (0, comm.size):
            raise RuntimeError(f"MPI disagreement in quasi-steady stopping criterion: {steady_votes}/{comm.size} ranks requested termination.")

        if steady_votes == comm.size:
            print(f" •QUASI-STEADY STATE REACHED at t={actTime:.6e} | steady_error={steady_error:.3e} < tol={cfg.steady_tol:.3e} | phi={steady_err_phi:.3e}, psiF={steady_err_psiF:.3e}, psiA={steady_err_psiA:.3e}, psiP={steady_err_psiP:.3e}, Vb={steady_err_Vb:.3e}, Wb={steady_err_Wb:.3e} | successful checks={steady_success_count}/{cfg.steady_required_checks} | MPI ranks={steady_votes}/{comm.size}")
            break

# =============================================================================
# 11. FINAL CHECKPOINT AND PLOTS
# =============================================================================
if comm.rank == 0 and len(Wb_time_list) > 1:
    plot_timeseries(np.array(Wb_time_list), np.array(Wb_vals), ylabel=r"$W_{\mathrm{b}}$", title=r"Extracellular Wnt $W_{\mathrm{b}}(t)$",
                    filename=Wb_plot, color="tab:orange", x_limits=(0, cfg.finalT), y_limits=None)
    plot_timeseries(np.array(Vb_time_list), np.array(Vb_vals), ylabel=r"$V_{\mathrm{b}}$", title=r"Intracellular receptors $V_{\mathrm{b}}(t)$",
                    filename=Vb_plot, color="tab:blue", x_limits=(0, cfg.finalT), y_limits=None)

if cfg.outputFlag:
    skip_final_write = last_vtk_write_time is not None and np.isclose(actTime, last_vtk_write_time, rtol=0.0, atol=1e-14)

    if skip_final_write:
        print(" --• Skipping final VTK output -- already written by the most recent periodic snapshot at the same time")
    else:
        print(" --• Writing final VTK output")
        gridView.writeVTK(vtkOutput, pointdata=[psiF_h, psiA_h, psiP_h, muP_h, phi_h, muPhi_h], celldata=[partitionFunction(gridView)], number=visCounter)

        if comm.rank == 0:
            final_dt = last_accepted_dt if last_accepted_dt is not None else cfg.dt.value   # fallback only if no step was ever accepted this run
            vtk_meta[visCounter] = {"time": actTime, "dt": final_dt, "W_b": float(W_b_const.value), "V_b": float(V_b_const.value), "steps": steps, "accepted_steps": accepted_steps,
                                    "beta_bump": cfg.beta_bump, "k_phs": cfg.k_phs, "k_int": cfg.k_int}
            save_results_json(vtk_meta, vtk_meta_path)

        visCounter += 1

    comm.barrier()

if cfg.checkpoint_flag:
    save_checkpoint(gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, cfg.dt.value, steps, visCounter, accepted_steps,  # io_utils.py
                    prev_refineTol, prev_coarsenTol, z, M_R, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_const, checkpoint_path)