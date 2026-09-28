# =============================================================================
# mesh_setup.py
# Grid loading, volume computation, initial conditions, and checkpoint loading
# for the wntMian.py 
# =============================================================================

import os
import sys
import json
import numpy as np
import meshio
from ufl import *
from dune.ufl import *
import dune.common.pickle   #type:ignore
from dune.common import comm   #type:ignore
from dune.fem.space import lagrange
from dune.fem.function import gridFunction
from dune.alugrid import aluConformGrid as leafGridView
from dune.grid import reader
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from gmsh2dgf import gmsh2DGF
from config import cfg
from functools import partial
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None

# =============================================================================
# 1. GRID UTILITIES
# =============================================================================
def compute_h_avg(gridView):
    """Compute global average edge length across all MPI ranks."""
    local_h_sum = 0.0
    local_h_count = 0
    for e in gridView.elements:
        geo = e.geometry
        v0 = np.array(geo.corner(0))
        v1 = np.array(geo.corner(1))
        v2 = np.array(geo.corner(2))
        local_h_sum += np.linalg.norm(v1 - v0) + np.linalg.norm(v2 - v1) + np.linalg.norm(v0 - v2)
        local_h_count += 3
    return float(comm.sum(local_h_sum)) / float(comm.sum(local_h_count))

def update_dt_bounds_from_mesh(gridView):
    """Recompute dt_min and dt_max from current mesh h_avg. Returns (dt_min, dt_max)."""
    h = compute_h_avg(gridView)
    dt_max = 3e3 * (h**2)
    dt_min = 1e-5 * (h**2)
    print(f" •dt bounds updated from mesh: h_avg={h:.4e} --> dt_min={dt_min:.4e}, dt_max={dt_max:.4e}")
    return dt_min, dt_max

def compute_volumes(gridView, bbox_margin):
    """Compute interior volume via divergence theorem and exterior volume via bounding box."""
    local_vol_in = 0.0
    for e in gridView.elements:
        geo = e.geometry
        v0 = np.array(geo.corner(0))
        v1 = np.array(geo.corner(1))
        v2 = np.array(geo.corner(2))
        center = (v0 + v1 + v2) / 3.0
        normal = np.cross(v1 - v0, v2 - v0)  # area-weighted outward normal
        local_vol_in += np.dot(center, normal)
    vol_in = float(comm.sum(local_vol_in)) / 6.0

    coords = np.array(gridView.coordinates())
    local_min = coords.min(axis=0) if len(coords) > 0 else np.zeros(3)
    local_max = coords.max(axis=0) if len(coords) > 0 else np.zeros(3)
    global_min = np.array([comm.min(float(local_min[i])) for i in range(3)])
    global_max = np.array([comm.max(float(local_max[i])) for i in range(3)])
    box_min = global_min - bbox_margin
    box_max = global_max + bbox_margin
    sides = box_max - box_min  # [Lx, Ly, Lz]
    vol_box = float(np.prod(sides))
    vol_ext = vol_box - vol_in
    outerBd3D = 2 * float(sides[0]*sides[1] + sides[1]*sides[2] + sides[0]*sides[2])  # surface area of box
    print(f" •vol_in={vol_in:.4e} | vol_box={vol_box:.4e} | vol_ext={vol_ext:.4e} | outerBd3D={outerBd3D:.4e}")
    return vol_in, vol_ext, outerBd3D

def rand_initial(space, lowest_val, highest_val, name, seed=None):
    """
    Fill a discrete function with uniform random values.
    Each rank gets seed + comm.rank so the random field is genuinely different across partitions.
    seed=None gives fully random (non-reproducible) initialisation.
    """
    to_random = space.interpolate(0, name=name)
    arr = to_random.as_petsc.getArray()    # Access DOFs of to_random as a NumPy array (direct view of DOFs)
    rng_seed = None if seed is None else seed + comm.rank
    rng = np.random.default_rng(rng_seed)  # Create a random number generator
    arr[:] = rng.uniform(lowest_val, highest_val, size=arr.shape[0])  # Fill in the DOFs array with these random numebrs (modify in-place)
    to_random.as_petsc.assemble()  # Finalise the PETSc vector after modification
    return to_random

def init_adapt_thresholds(phi_h, psiP_h, muP_h, muPhi_h, gridView):
    """
    Computes initial global refinement and coarsening thresholds based on the initial indicator.
    """
    indicator, _ = build_indicator(phi_h, psiP_h, muP_h, muPhi_h, gridView)
    _vals = np.array([indicator(e, e.geometry.center) for e in gridView.elements])
    local_max = float(np.max(_vals)) if len(_vals) > 0 else 0.0
    global_max = comm.max(local_max)
    return cfg.raw_ref_frac * global_max, cfg.raw_crs_frac * global_max

def build_indicator(phi_h, psiP_h, muP_h, muPhi_h, gridView):
    """
    Builds the AMR indicator ONCE. Returns the indicator together with the four
    Constant objects holding the per-field normalization maxima, so the caller
    can refresh those maxima each AMR cycle via update_indicator_maxima(...)
    WITHOUT rebuilding the indicator expression itself.
    """
    # Raw gradient magnitudes -- live UFL expressions, always reflect current field state
    g_phi = dot(grad(phi_h), grad(phi_h))
    g_psiP = dot(grad(psiP_h), grad(psiP_h))
    g_muP = dot(grad(muP_h), grad(muP_h))
    g_muPhi = dot(grad(muPhi_h), grad(muPhi_h))

    # Temporary grid functions to evaluate maxima (reused each refresh, not rebuilt)
    tmp_phi = gridFunction(g_phi, gridView, order=1)
    tmp_psiP = gridFunction(g_psiP, gridView, order=1)
    tmp_muP = gridFunction(g_muP, gridView, order=1)
    tmp_muPhi = gridFunction(g_muPhi, gridView, order=1)

    # Initial maxima, stored as Constants so they can be updated in place later
    max_phi_c = Constant(_compute_max(tmp_phi, gridView), name="max_phi")
    max_psiP_c = Constant(_compute_max(tmp_psiP, gridView), name="max_psiP")
    max_muP_c = Constant(_compute_max(tmp_muP, gridView), name="max_muP")
    max_muPhi_c = Constant(_compute_max(tmp_muPhi, gridView), name="max_muPhi")

    # Normalise each field by its Constant max -- since these are Constants (not
    # plain floats), updating .value later automatically updates the indicator
    # expression too, with no need to rebuild it.
    n_phi = g_phi / max_value(max_phi_c, as_ufl(1e-30))
    n_psiP = g_psiP / max_value(max_psiP_c, as_ufl(1e-30))
    n_muP = g_muP / max_value(max_muP_c, as_ufl(1e-30))
    n_muPhi = g_muPhi / max_value(max_muPhi_c, as_ufl(1e-30))

    indicator_expr = max_value(max_value(max_value(n_phi, n_psiP), n_muP), n_muPhi)
    indicator = gridFunction(indicator_expr, gridView, name="indicator", order=1)

    # Bundle the temp grid functions and Constants together so the caller can refresh them
    maxima = {
        "tmp_phi": tmp_phi, "tmp_psiP": tmp_psiP, "tmp_muP": tmp_muP, "tmp_muPhi": tmp_muPhi,
        "max_phi_c": max_phi_c, "max_psiP_c": max_psiP_c, "max_muP_c": max_muP_c, "max_muPhi_c": max_muPhi_c,
    }
    return indicator, maxima


def _compute_max(tmp_field, gridView):
    """Helper: compute the global max of a grid function over all elements."""
    local_max = float(max((tmp_field(e, e.geometry.center) for e in gridView.elements), default=0.0))
    return comm.max(local_max)


def update_indicator_maxima(maxima, gridView):
    """
    Call this every AMR cycle, BEFORE evaluating the indicator, to refresh the
    normalization maxima against the CURRENT field state. This is the fix for
    the frozen-denominator bug -- no expression rebuild needed, just updates
    the Constants in place.
    """
    maxima["max_phi_c"].value = _compute_max(maxima["tmp_phi"], gridView)
    maxima["max_psiP_c"].value = _compute_max(maxima["tmp_psiP"], gridView)
    maxima["max_muP_c"].value = _compute_max(maxima["tmp_muP"], gridView)
    maxima["max_muPhi_c"].value = _compute_max(maxima["tmp_muPhi"], gridView)

def count_owned_dofs(phi_h):
    """Count owned DOFs only. Excludes ghosts, no double counting."""
    return int(comm.sum(phi_h.as_petsc.local_size))

def owned_dof_distribution(phi_h):
    """Return local/global owned-DOF distribution across MPI ranks."""
    local_dofs = int(phi_h.as_petsc.local_size)

    min_dofs = int(comm.min(local_dofs))
    max_dofs = int(comm.max(local_dofs))
    total_dofs = int(comm.sum(local_dofs))
    avg_dofs = total_dofs / comm.size

    imbalance = max_dofs / max(avg_dofs, 1.0)
    min_fraction = min_dofs / max(avg_dofs, 1.0)

    return local_dofs, min_dofs, avg_dofs, max_dofs, total_dofs, imbalance, min_fraction

# =============================================================================
# 2. ADAPTIVE TIMESTEPPING
# =============================================================================
def adaptive_timestep(psiF_h, psiF_n, psiA_h, psiA_n, psiP_h, psiP_n, phi_h, phi_n, dt, deltaT, dt_min, dt_max):
    """
    Adjusts the time step size dynamically by monitoring solution field changes, enforcing rejection thresholds and growth/shrinkage constraints.
    """
    n_psiF = psiF_h.as_petsc.local_size
    n_psiA = psiA_h.as_petsc.local_size
    n_psiP = psiP_h.as_petsc.local_size
    n_phi = phi_h.as_petsc.local_size

    change_psiF = comm.max(float(np.max(np.abs(psiF_h.as_petsc.getArray()[:n_psiF] - psiF_n.as_petsc.getArray()[:n_psiF]))))
    change_psiA = comm.max(float(np.max(np.abs(psiA_h.as_petsc.getArray()[:n_psiA] - psiA_n.as_petsc.getArray()[:n_psiA]))))
    change_psiP = comm.max(float(np.max(np.abs(psiP_h.as_petsc.getArray()[:n_psiP] - psiP_n.as_petsc.getArray()[:n_psiP]))))
    change_phi = comm.max(float(np.max(np.abs(phi_h.as_petsc.getArray()[:n_phi] - phi_n.as_petsc.getArray()[:n_phi]))))
    max_change = max(change_phi, change_psiF, change_psiA, change_psiP)  # Global max change across all fields

    if max_change > cfg.max_change_allowed:
        if deltaT <= dt_min * (1.0 + 1e-12):
            raise RuntimeError(
                f"Solution-change criterion failed at dt_min={dt_min:.3e}. "
                f"max_change={max_change:.3e} but allowed={cfg.max_change_allowed:.3e}. "
                f"change_phi={change_phi:.3e}, change_psiF={change_psiF:.3e}, "
                f"change_psiA={change_psiA:.3e}, change_psiP={change_psiP:.3e}. Aborting."
            )

        reduction_pct = (1 - cfg.dt_reduce_after_change_reject) * 100
        print(f" •REJECT STEP. Max change allowed={cfg.max_change_allowed:.3e} but:")
        print(f"   ---- change_phi={change_phi:.3e} | change_psiF={change_psiF:.3e} | change_psiA={change_psiA:.3e} | change_psiP={change_psiP:.3e}")
        print(f" •dt={dt.value:.3e} will be reduced by {reduction_pct:.2f}% until it hits dt_min={dt_min:.3e}")
        dt.value = max(deltaT * cfg.dt_reduce_after_change_reject, dt_min)
        print(f" •dt reduced! dt={dt.value:.3e} >= dt_min={dt_min:.3e}")
        return dt.value, True

    # In case there is no rejection, we choose different admissible time steps based on the change of (psiP, phi) solutions ONLY
    # Admissible time steps: Scale current dt by how far we are from the desired change chosen earlier, then pull back by safety_factor
    # If teh change of a solution is 0, just use dt_max. No need to be conservative when nothing is changing

    dt_psiF = dt.value * cfg.safety_factor * (cfg.target_change / change_psiF) if change_psiF > 1e-15 else dt_max
    dt_psiA = dt.value * cfg.safety_factor * (cfg.target_change / change_psiA) if change_psiA > 1e-15 else dt_max
    dt_psiP = dt.value * cfg.safety_factor * (cfg.target_change / change_psiP) if change_psiP > 1e-15 else dt_max
    dt_phi = dt.value * cfg.safety_factor * (cfg.target_change_phi / change_phi) if change_phi > 1e-15 else dt_max

    dt_new = min(dt_psiF, dt_psiA, dt_psiP, dt_phi)  # Take the most conservative suggestion
    dt_new = max(dt_new, dt.value * cfg.dt_shrink_rate)  # dt cannot shrink faster than dt_shrink_rate
    dt_new = min(dt_new, dt.value * cfg.dt_growth_rate)  # dt cannot grow faster than dt_growth_rate
    dt.value = float(np.clip(dt_new, dt_min, dt_max))  # Enforce hard bounds
    return dt.value, False

# =============================================================================
# 3. GRID SETUP AND INITIAL CONDITIONS
# =============================================================================
def build_grid(mesh_type, script_dir):
    # Build everything from scratch (if you add/use a new mesh type, this where you include and build it seperately!)
    if mesh_type == "sphere": # Note this only takes .msh files
        mesh_path = os.path.join(script_dir, "meshes", "sphere.msh")  # NAME OF MESH GIVEN HERE
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        mesh = meshio.read(mesh_path) # Read the .msh file
        points = mesh.points
        norms = np.linalg.norm(points, axis=1, keepdims=True)  # Norm of each point. "norms" has shape (N = number of points,1)
        points = points / norms  # Project onto unit sphere
        cells = {"triangle": mesh.cells_dict["triangle"]}
        dgf_str = gmsh2DGF(points, cells, dim=3)
        alu_params = (
            "\nGridParameter\n"
            "  bisectioncompatibility 0\n"
            "  refinementedge ARBITRARY\n"
            "#\n" )
        dgf_str = dgf_str.rstrip().rstrip('#').rstrip() + "\n" + alu_params + "#\n"
        domain = (reader.dgfString, dgf_str)
        gridView = adaptiveGridView(leafGridView(domain, dimgrid=2, dimworld=3, lbMethod=9))
        gridView.hierarchicalGrid.globalRefine(1) # initial refinement
        gridView.hierarchicalGrid.loadBalance()
    elif mesh_type == "cell": # Note this only takes .dgf files
        mesh_path = os.path.join(script_dir, "meshes", "cell.dgf")  # NAME OF MESH GIVEN HERE
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"DGF file not found: {mesh_path}")
        with open(mesh_path, "r") as f:
            dgf_str = f.read()
        alu_params = (
            "\nGridParameter\n"
            "  bisectioncompatibility 0\n"
            "  refinementedge ARBITRARY\n"
            "#\n")
        dgf_str = dgf_str.rstrip().rstrip('#').rstrip() + "\n" + alu_params + "#\n"
        domain = (reader.dgfString, dgf_str)  # same as sphere
        gridView = adaptiveGridView(leafGridView(domain, dimgrid=2, dimworld=3, lbMethod=9))
        gridView.hierarchicalGrid.globalRefine(1)
        gridView.hierarchicalGrid.loadBalance()
    else:
        raise ValueError(f"Unknown mesh_type='{mesh_type}'. Choose from: 'sphere', 'cell'")
    return gridView

def build_initial_conditions(gridView, pOrder):
    space = lagrange(gridView, order=pOrder, storage='petsc')

    # =========================================================================
    # Resting-state initial condition
    # =========================================================================
    if cfg.InitialCondition == "resting":

        # Small reproducible perturbation around the prescribed mean lipid composition
        rnd_phi = 0.01
        phi_h = rand_initial(space, cfg.phi_bar - rnd_phi, cfg.phi_bar + rnd_phi, name="phi_h", seed=0)
        muPhi_h = space.interpolate(0.0, name="muPhi_h")

        # No-Wnt receptor equilibrium:
        # k_ads*V_b*(1-psiF) = k_des*psiF
        psiF_eq = (cfg.k_ads * cfg.V_b_init/ (cfg.k_ads * cfg.V_b_init + cfg.k_des))
        psiF_h = space.interpolate(psiF_eq, name="psiF_h")
        psiA_h = space.interpolate(0.0, name="psiA_h")
        psiP_h = space.interpolate(0.0, name="psiP_h")

        muP_h = space.interpolate(0.0, name="muP_h")

        print(
            f" •Resting IC: phi in "
            f"[{cfg.phi_bar-rnd_phi:.4f},{cfg.phi_bar+rnd_phi:.4f}], "
            f"psiF_eq={psiF_eq:.8f}, psiA=0, psiP=0")

    # =========================================================================
    # Original/default initial condition
    # =========================================================================
    elif cfg.InitialCondition == "default":

        rnd_phi = 0.01
        phi_h = rand_initial(space, cfg.phi_bar - rnd_phi, cfg.phi_bar + rnd_phi, name="phi_h", seed=0)
        muPhi_h = space.interpolate(0.0, name="muPhi_h")
        psiF_h = space.interpolate(0.2, name="psiF_h")
        psiA_h = space.interpolate(0.05, name="psiA_h")

        rnd_p = 5e-5
        psiP_h = rand_initial(space, 5e-4 - rnd_p, 5e-4 + rnd_p, name="psiP_h", seed=0)
        muP_h = space.interpolate(0.0, name="muP_h")
    else:
        raise ValueError(
            f"Unknown cfg.InitialCondition={cfg.InitialCondition!r}. "
            "Choose from: 'default', 'resting'.")

    return phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h
# =============================================================================
# 4. CHECKPOINT LOAD
# =============================================================================
def _load_checkpoint_fields(checkpoint_path):
    meta_path = f"{checkpoint_path}.meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    saved_with_ranks = meta["saved_with_ranks"]
    if saved_with_ranks != comm.size:
        raise RuntimeError(
            f"Checkpoint was saved with {saved_with_ranks} MPI rank(s) but this run is using "
            f"{comm.size}. DUNE checkpoints must be restored with the SAME rank count they were "
            f"saved with. Resubmit with the matching rank count to resume."
        )

    rank_checkpoint_path = f"{checkpoint_path}.rank{comm.rank}"
    file_exists = os.path.exists(rank_checkpoint_path)
    all_exist = comm.min(int(file_exists))
    if not all_exist:
        if not file_exists:
            sys.stderr.write(f" • WARNING [rank {comm.rank}]: missing checkpoint file: {rank_checkpoint_path}\n")
            sys.stderr.flush()
        comm.barrier()
        raise RuntimeError("One or more per-rank checkpoint files are missing -- see per-rank WARNING lines above.")
    with open(rank_checkpoint_path, "rb") as f:
        gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, state = dune.common.pickle.load(f)  # Fill in

    return gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, state, rank_checkpoint_path

def load_checkpoint_resume(checkpoint_path):
    gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, state, rank_checkpoint_path = _load_checkpoint_fields(checkpoint_path)

    physical_parameters_changed = False

    actTime = state["actTime"]
    dt_val = state["dt"]
    steps = state["steps"]
    visCounter = state["visCounter"]
    accepted_steps = state["accepted_steps"]
    prev_refineTol = state["prev_refineTol"]
    prev_coarsenTol = state["prev_coarsenTol"]
    z = np.array(state["z"])  # Recovers the ODE solution from previous simulation
    Wb_time_list = state["Wb_time_list"]  # Recovers the time points of the ODE solution from previous simulation
    Wb_vals = state["Wb_vals"]  # Recovers the values of the ODE solution recorded at each time point from previous simulation
    Vb_time_list = state["Vb_time_list"]
    Vb_vals  = state["Vb_vals"]
    V_b_resume = state["V_b"]
    M_R = state["M_R"]

    saved_cfg = state.get("cfg_snapshot")
    if saved_cfg is None:
        raise RuntimeError(
            "Checkpoint does not contain a cfg snapshot, so parameter consistency "
            "cannot be verified safely for resume. Use load_mode='ic_only'."
        )
    current_cfg = cfg.snapshot()

    block_on_change = {
        "mesh_type", "bbox_margin", "pOrder",
    }

    physical_on_change = {
        "beta_bump",
        "k_B", "theta", "sgm", "alph", "xi_a", "xi_p", "chi_p", "lmbd",
        "L_star", "t_star", "eps", "eta_ld", "eta_lo", "eta_int", "eta_ext",
        "r_r", "e_m",
        "k_ads", "k_des", "k_on", "k_off", "k_phs", "k_dph", "k_int", "k_phi",
        "phi_bar", "gamma", "m_0", "psiP_star",
        "eps_star", "D_star", "K_phi", "K_eps", "N_p", "X_p", "Lambda",
        "Pi_a_val", "Pi_p_val",
    }

    warn_on_change = {
        "use_adaptive_mesh", "adaptFreq", "minLevel", "maxLevel",
        "max_tol_change", "raw_ref_frac", "raw_crs_frac",
        "use_adaptive_dt", "target_change", "safety_factor",
        "max_amr_allowed", "max_change_allowed",
        "dt_reduce_after_change_reject", "dt_reduce_after_picard_reject",
        "dt_reduce_after_newton_reject", "dt_reduce_after_amr_change",
        "dt_shrink_rate", "dt_growth_rate",
        "newtonTol", "newtonMaxIt", "linear_method", "preconditioner",
        "linearMaxIt", "gmresRestart", "picardTol", "picardMaxIt",
    }

    blocked_diffs = []
    physical_diffs = []
    warning_diffs = []

    for name, saved_value in saved_cfg.items():
        if name not in current_cfg:
            continue

        current_value = current_cfg[name]

        if isinstance(saved_value, (int, float)) and isinstance(current_value, (int, float)):
            equal = np.isclose(saved_value, current_value, rtol=1e-12, atol=1e-15)
        else:
            equal = saved_value == current_value

        if equal:
            continue

        difference = f"{name}: saved={saved_value!r} --> current={current_value!r}"

        if name in block_on_change:
            blocked_diffs.append(difference)
        elif name in physical_on_change:
            physical_diffs.append(difference)
        elif name in warn_on_change:
            warning_diffs.append(difference)

    if blocked_diffs:
        text = "\n".join(f"     {item}" for item in blocked_diffs)
        raise RuntimeError(
            "Cannot resume because structurally incompatible checkpoint parameters changed:\n"
            f"{text}\n"
            "Restore the saved values or use load_mode='ic_only'."
        )

    if physical_diffs:
        physical_parameters_changed = True
        print(" • WARNING: physical/model parameters differ from the saved run.")
        print(" • Continuing as a parameter-switch experiment from the checkpoint state:")
        for item in physical_diffs:
            print(f"     {item}")

    if warning_diffs:
        print(" • WARNING: numerical-control parameters differ from the saved run:")
        for item in warning_diffs:
            print(f"     {item}")

    if cfg.finalT < actTime:
        raise RuntimeError(
            f"Cannot resume: cfg.finalT={cfg.finalT} is smaller than "
            f"checkpoint actTime={actTime}."
        )

    cfg.dt.value = dt_val

    print(f"Checkpoint loaded from: {checkpoint_path}.rank<N> (one file per rank; rank 0 shown: {rank_checkpoint_path})")
    print(f" Resuming from actTime={actTime:.4e}, step={steps}, visCounter={visCounter}, dt={dt_val:.4e}, W_b={z[0]:.4e}")

    return (gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, dt_val, steps, visCounter, accepted_steps,
            prev_refineTol, prev_coarsenTol, z, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_resume, M_R, physical_parameters_changed)

def load_checkpoint_ic_only(checkpoint_path):
    gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, state, rank_checkpoint_path = _load_checkpoint_fields(checkpoint_path)

    actTime = 0.0
    steps = 0
    visCounter = 0
    accepted_steps = 0
    z = np.array([cfg.W_b_init])  # Will be used to store ODE solution. Currently holds initial value
    Wb_time_list = []
    Wb_vals = []
    Vb_time_list = []
    Vb_vals = []
    dt_val = None
    V_b_resume = cfg.V_b_init
    M_R = None
    prev_refineTol = None
    prev_coarsenTol = None
    physical_parameters_changed = False
    print(f"Checkpoint loaded as IC only from: {checkpoint_path}.rank<N> (one file per rank; rank 0 shown: {rank_checkpoint_path})")

    return (gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, dt_val, steps, visCounter, accepted_steps,
            prev_refineTol, prev_coarsenTol, z, Wb_time_list, Wb_vals, Vb_time_list, Vb_vals, V_b_resume, M_R, physical_parameters_changed)


def setup_fresh(script_dir, mesh_type, pOrder):
    
    gridView = build_grid(mesh_type, script_dir)
    (phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h) = build_initial_conditions(gridView, pOrder)
    M_R_initial = None
    V_b_initial = cfg.V_b_init

    actTime = 0.0
    steps = 0
    visCounter = 0
    accepted_steps = 0

    z = np.array([cfg.W_b_init])

    Wb_time_list = []
    Wb_vals = []
    Vb_time_list = []
    Vb_vals = []

    physical_parameters_changed = False

    return (gridView, phi_h, muPhi_h, psiF_h, psiA_h, psiP_h, muP_h, actTime, steps, visCounter, accepted_steps, z, Wb_time_list, Wb_vals,
        Vb_time_list, Vb_vals, V_b_initial, M_R_initial, physical_parameters_changed)