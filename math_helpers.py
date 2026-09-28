# =============================================================================
# math_helpers.py
# Mathematical helper functions for the wntMain.py and wntBench.py.
# =============================================================================

from petsc4py import PETSc   # type: ignore
from ufl import *
from dune.ufl import *
from config import cfg
from dune.common import comm   #type:ignore
from functools import partial
from dune.fem import integrate
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None
import numpy as np


# =============================================================================
# 1. PHI-DEPENDENT DIFFUSIVITIES AND DRIFT
# =============================================================================
# These depend on eta_s(phi) via Saffman-Delbruck formula
eta_s = lambda phi: conditional(le(phi, -1.0), cfg.eta_ld, conditional(ge(phi, 1.0), cfg.eta_lo, (cfg.eta_lo + cfg.eta_ld)/2 + (cfg.eta_lo - cfg.eta_ld)/4 * phi * (3 - phi**2)))
ell_sd = lambda phi: eta_s(phi) / (cfg.eta_int + cfg.eta_ext)   # Saffman-Delbruck length [nm]

eta = lambda phi: (4 * pi * eta_s(phi)) / (ln(2 * ell_sd(phi) / cfg.r_r) - cfg.e_m)   # Receptors drag coefficient for free receptors [pN s / nm]

delt = lambda phi: (as_ufl(cfg.k_B * cfg.theta)) / (eta(phi) * as_ufl(cfg.D_star))   # delta(phi): nondimensional membrane diffusivity 
A_a = lambda phi: (as_ufl(cfg.xi_a) / (eta(phi) * as_ufl(cfg.D_star)))   # A_a(phi): drift strength for psiA

# =============================================================================
# 2. MOBILITIES, POTENTIALS, AND AUXILIARY FUNCTIONS
# =============================================================================
# rho_r function
rhoR = lambda yf, ya, yp:  yf + ya + yp

# Mobilities 
M_phi = lambda phi: cfg.m_0 *(1- phi**2)
Theta_lambda_n = lambda q: max_value(q, as_ufl(cfg.lambda_n))
M_p = lambda y_p, y_rho, y_phi: delt(y_phi) * y_p * Theta_lambda_n(1-y_rho) * exp(-y_p/cfg.psiP_star)

# Double-well potential for Cahn-Hilliard
W = lambda y_phi: (1/4) * (y_phi**2 - 1)**2 # not used
dW = lambda y_phi: (y_phi**3 - y_phi) # not used
# W = W_c - W_e  # Convex splitting
Wc = lambda y_phi: (1/4) * y_phi**4 + (1/2) * y_phi**2 # not used
We = lambda y_phi: y_phi**2 - 1/4 # not used
dWc = lambda y_phi: y_phi**3 + y_phi
dWe = lambda y_phi: as_ufl(2.0)*y_phi

# h_beta: switch bump function
def h_bet(y_phi, beta=None):
    beta = beta if beta is not None else cfg.beta_bump
    u = (y_phi - as_ufl(1.0)) / as_ufl(beta)
    u2 = u**2
    expr = (as_ufl(1.0) - u2)**3
    return conditional(lt(u2, 1.0), expr, as_ufl(0.0))


def build_lumped_mass_weights(mass_matrix):
    """Build the lumped mass weights m=M*1 used by the mass-conserving bound correction."""
    M = mass_matrix.as_petsc
    ones = M.createVecRight()
    weights = M.createVecLeft()
    ones.set(1.0)
    ones.assemble()
    M.mult(ones, weights)
    weights.assemble()
    ones.destroy()
    return weights


def mass_conserving_bound_correction(phi_h, lumped_mass_weights, lower=-1.0, upper=1.0, tol=1e-13, maxiter=100):
    """Project phi_h onto [lower,upper] while preserving its lumped finite-element mass."""
    phi_vec = phi_h.as_petsc
    n_owned = phi_vec.local_size
    phi_old = phi_vec.getArray(readonly=True)[:n_owned].copy()
    weights = lumped_mass_weights.getArray(readonly=True)[:n_owned].copy()

    local_min = float(np.min(phi_old)) if phi_old.size > 0 else upper
    local_max = float(np.max(phi_old)) if phi_old.size > 0 else lower
    phi_min_before = float(comm.min(local_min))
    phi_max_before = float(comm.max(local_max))

    local_corrected = int(np.count_nonzero((phi_old < lower) | (phi_old > upper)))
    corrected_dofs = int(comm.sum(local_corrected))

    target_mass = float(comm.sum(float(np.dot(weights, phi_old))))
    total_weight = float(comm.sum(float(np.sum(weights))))
    minimum_mass = lower * total_weight
    maximum_mass = upper * total_weight
    mass_tol = tol * max(1.0, abs(target_mass), total_weight)

    if target_mass < minimum_mass - mass_tol or target_mass > maximum_mass + mass_tol:
        raise RuntimeError(f"Mass-conserving bound correction is infeasible: target_mass={target_mass:.16e}, admissible=[{minimum_mass:.16e},{maximum_mass:.16e}].")

    if corrected_dofs == 0:
        return {"corrected": False, "corrected_dofs": 0, "lambda": 0.0, "phi_min_before": phi_min_before, "phi_max_before": phi_max_before, "phi_min_after": phi_min_before, "phi_max_after": phi_max_before, "mass_before": target_mass, "mass_after": target_mass, "mass_error": 0.0}

    lam_left = lower - phi_max_before
    lam_right = upper - phi_min_before
    lam_mid = 0.0

    for _ in range(maxiter):
        lam_mid = 0.5 * (lam_left + lam_right)
        corrected_trial = np.clip(phi_old + lam_mid, lower, upper)
        trial_mass = float(comm.sum(float(np.dot(weights, corrected_trial))))

        if abs(trial_mass - target_mass) <= mass_tol:
            break

        if trial_mass < target_mass:
            lam_left = lam_mid
        else:
            lam_right = lam_mid
    else:
        raise RuntimeError(f"Mass-conserving bound correction failed to converge after {maxiter} iterations.")

    phi_corrected = np.clip(phi_old + lam_mid, lower, upper)
    arr = phi_vec.getArray()
    arr[:n_owned] = phi_corrected
    phi_vec.assemble()
    _ghost_update(phi_vec)

    corrected_mass = float(comm.sum(float(np.dot(weights, phi_corrected))))
    local_min_after = float(np.min(phi_corrected)) if phi_corrected.size > 0 else upper
    local_max_after = float(np.max(phi_corrected)) if phi_corrected.size > 0 else lower
    phi_min_after = float(comm.min(local_min_after))
    phi_max_after = float(comm.max(local_max_after))
    mass_error = corrected_mass - target_mass

    if abs(mass_error) > 10.0 * mass_tol:
        raise RuntimeError(f"Mass-conserving bound correction produced mass_error={mass_error:.16e}.")

    return {"corrected": True, "corrected_dofs": corrected_dofs, "lambda": lam_mid, "phi_min_before": phi_min_before, "phi_max_before": phi_max_before, "phi_min_after": phi_min_after, "phi_max_after": phi_max_after, "mass_before": target_mass, "mass_after": corrected_mass, "mass_error": mass_error}

def _project_to_receptor_simplex(Y):
    """
    Row-wise Euclidean projection onto {x >= 0, x_1 + x_2 + x_3 <= 1}.
    """
    X = np.maximum(Y, 0.0)
    row_sum = np.sum(X, axis=1)
    bad = row_sum > 1.0
    if not np.any(bad):
        return X

    V = X[bad]
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    j = np.arange(1, 4)
    cond = U - cssv / j > 0.0
    rho = np.sum(cond, axis=1) - 1
    theta = cssv[np.arange(V.shape[0]), rho] / (rho + 1.0)
    X[bad] = np.maximum(V - theta[:, None], 0.0)

    return X


def mass_conserving_simplex_correction(psiF_h, psiA_h, psiP_h, lumped_mass_weights, tol=1e-13, maxiter=100):
    """
    Project (psiF, psiA, psiP) onto {psiF >= 0, psiA >= 0, psiP >= 0, psiF + psiA + psiP <= 1}
    while preserving the total lumped receptor mass.
    """

    n_owned = psiF_h.as_petsc.local_size
    psiF_old = psiF_h.as_petsc.getArray(readonly=True)[:n_owned].copy()
    psiA_old = psiA_h.as_petsc.getArray(readonly=True)[:n_owned].copy()
    psiP_old = psiP_h.as_petsc.getArray(readonly=True)[:n_owned].copy()
    weights = lumped_mass_weights.getArray(readonly=True)[:n_owned].copy()
    Y = np.column_stack((psiF_old, psiA_old, psiP_old))
    rho_old = np.sum(Y, axis=1)
    local_min = float(np.min(Y)) if Y.size > 0 else 0.0
    local_rho_max = float(np.max(rho_old)) if rho_old.size > 0 else 0.0
    psi_min_before = float(comm.min(local_min))
    rho_max_before = float(comm.max(local_rho_max))
    local_corrected = int(np.count_nonzero(
        (np.min(Y, axis=1) < 0.0) | (rho_old > 1.0)
    )) if Y.shape[0] > 0 else 0

    corrected_dofs = int(comm.sum(local_corrected))
    target_mass = float(comm.sum(float(np.dot(weights, rho_old))))
    total_weight = float(comm.sum(float(np.sum(weights))))
    mass_tol = tol * max(1.0, abs(target_mass), total_weight)
    if target_mass < -mass_tol or target_mass > total_weight + mass_tol:
        raise RuntimeError(
            f"Mass-conserving simplex correction is infeasible: "
            f"target_mass={target_mass:.16e}, admissible=[0,{total_weight:.16e}]."
        )

    if corrected_dofs == 0:
        return {
            "corrected": False,
            "corrected_dofs": 0,
            "lambda": 0.0,
            "psi_min_before": psi_min_before,
            "rho_max_before": rho_max_before,
            "psi_min_after": psi_min_before,
            "rho_max_after": rho_max_before,
            "mass_before": target_mass,
            "mass_after": target_mass,
            "mass_error": 0.0,
        }

    max_abs = float(comm.max(float(np.max(np.abs(Y))) if Y.size > 0 else 0.0))
    lam_left = -(max_abs + 1.0)
    lam_right = max_abs + 1.0
    lam_mid = 0.0

    for _ in range(maxiter):
        lam_mid = 0.5 * (lam_left + lam_right)
        X = _project_to_receptor_simplex(Y + lam_mid)
        rho = np.sum(X, axis=1)
        mass_mid = float(comm.sum(float(np.dot(weights, rho))))
        if abs(mass_mid - target_mass) <= mass_tol:
            break
        if mass_mid < target_mass:
            lam_left = lam_mid
        else:
            lam_right = lam_mid
    else:
        raise RuntimeError(f"Mass-conserving simplex correction failed to converge after {maxiter} iterations.")        

    X = _project_to_receptor_simplex(Y + lam_mid)
    psiF_arr = psiF_h.as_petsc.getArray()
    psiA_arr = psiA_h.as_petsc.getArray()
    psiP_arr = psiP_h.as_petsc.getArray()
    psiF_arr[:n_owned] = X[:, 0]
    psiA_arr[:n_owned] = X[:, 1]
    psiP_arr[:n_owned] = X[:, 2]
    psiF_h.as_petsc.assemble()
    psiA_h.as_petsc.assemble()
    psiP_h.as_petsc.assemble()
    _ghost_update(psiF_h.as_petsc)
    _ghost_update(psiA_h.as_petsc)
    _ghost_update(psiP_h.as_petsc)
    rho_new = np.sum(X, axis=1)
    local_min_after = float(np.min(X)) if X.size > 0 else 0.0
    local_rho_max_after = float(np.max(rho_new)) if rho_new.size > 0 else 0.0
    psi_min_after = float(comm.min(local_min_after))
    rho_max_after = float(comm.max(local_rho_max_after))
    mass_after = float(comm.sum(float(np.dot(weights, rho_new))))
    mass_error = mass_after - target_mass

    if abs(mass_error) > 10.0 * mass_tol:
        raise RuntimeError(f"Mass-conserving simplex correction produced mass_error={mass_error:.16e}.")

    simplex_tol = 1e-12
    if psi_min_after < -simplex_tol or rho_max_after > 1.0 + simplex_tol:
        raise RuntimeError(
            f"Simplex correction failed: min(psi)={psi_min_after:.16e}, "
            f"rho_max={rho_max_after:.16e}.")
    
    return {
        "corrected": True,
        "corrected_dofs": corrected_dofs,
        "lambda": lam_mid,
        "psi_min_before": psi_min_before,
        "rho_max_before": rho_max_before,
        "psi_min_after": psi_min_after,
        "rho_max_after": rho_max_after,
        "mass_before": target_mass,
        "mass_after": mass_after,
        "mass_error": mass_error,
    }

def relative_max_change(u_h, u_ref):
    u_vec = u_h.as_petsc
    ref_vec = u_ref.as_petsc
    n_owned = u_vec.local_size
    u_arr = u_vec.getArray(readonly=True)[:n_owned]
    ref_arr = ref_vec.getArray(readonly=True)[:n_owned]
    local_diff = float(np.max(np.abs(u_arr-ref_arr))) if u_arr.size > 0 else 0.0
    local_ref = float(np.max(np.abs(ref_arr))) if ref_arr.size > 0 else 0.0
    diff = float(comm.max(local_diff))
    ref_max = float(comm.max(local_ref))
    return diff/max(1.0, ref_max)

def relative_L2_change(u_h, u_ref, gridView):
    diff = float(integrate((u_h-u_ref)**2, gridView, order=10))**0.5
    ref = float(integrate(u_ref**2, gridView, order=10))**0.5
    return diff/max(1.0, ref)

# =============================================================================
# 3. MPI GHOST UPDATE
# =============================================================================
def _ghost_update(vec):
    """
    Synchronizes parallel data by updating local ghost values with the latest DOF values from neighboring owning ranks. vec is a PETSc object.
    """
    try:
        vec.ghostUpdateBegin(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        vec.ghostUpdateEnd(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
    except PETSc.Error:
        pass
    # INSERT: received value replaces the ghost copy. Alternative ---> ADD: add the received value to current ghost copy. Used when accumulating contributions
    # FORWARD: communication goes in the direction: owner → ghost. Alternative ---> REVERSE: ghosts send their values back to the owning rank