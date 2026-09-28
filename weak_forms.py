# =============================================================================
# weak_forms.py
# Variational forms, operators, schemes, and Schur solver setup
# for the wntMain.py and wntBench.py.
# =============================================================================

from petsc4py import PETSc   # type: ignore
from ufl import *
from dune.ufl import *
from dune.fem.space import lagrange
from dune.fem.operator import galerkin as galerkinOperator
from dune.fem.scheme import galerkin as galerkinScheme
from dune.common import comm   #type:ignore
from solverMPI import SchurSolver
from math_helpers import *
from math_helpers import _ghost_update
from config import cfg
from functools import partial
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None

# =============================================================================
# 1. SPACE AND TRIAL/TEST FUNCTIONS
# =============================================================================
def build_space(gridView):  
    # Create FEM space
    space = lagrange(gridView, order=cfg.pOrder, storage='petsc')
    return space

def build_trial_test_functions(space):
    # Trial and Test functions
    psiF = TrialFunction(space); zetaF = TestFunction(space)
    psiA = TrialFunction(space); zetaA = TestFunction(space)
    psiP = TrialFunction(space); zetaP = TestFunction(space)
    phi = TrialFunction(space); zetaPhi = TestFunction(space)
    muP = TrialFunction(space); omgP = TestFunction(space)
    muPhi = TrialFunction(space); omgPhi = TestFunction(space)
    return (psiF, zetaF, psiA, zetaA, psiP, zetaP, phi, zetaPhi, muP, omgP, muPhi, omgPhi)

# =============================================================================
# 2. REACTION TERMS
# =============================================================================
def build_reaction_terms(psiF, psiA, psiP, phi, psiP_h, psiF_m, psiA_m, psiP_m, phi_n, W_b_const, V_b_const, phosphorylation_gate=None):

    max_val_psiP_0_m = max_value(psiP_m, as_ufl(0.0))
    # psiP_h is the accepted solution from the previous time step, used only in R_phi
    max_val_psiP_0_h = max_value(psiP_h, as_ufl(0.0))
    rhoR_f = rhoR(psiF, psiA_m, psiP_m)  # For psiF equation
    phs_gate = h_bet(phi_n) if phosphorylation_gate is None else phosphorylation_gate
    
    R_f = (cfg.k_ads * V_b_const * (1-rhoR_f) + cfg.k_off * psiA_m - (cfg.k_on * W_b_const + cfg.k_des) * psiF)
    R_a = (cfg.k_on * W_b_const * psiF_m + cfg.k_dph * max_val_psiP_0_m - cfg.k_off * psiA - cfg.k_phs * phs_gate * psiA)
    R_p = (cfg.k_phs * phs_gate * psiA_m - cfg.k_dph * psiP - cfg.k_int * h_bet(phi_n) * psiP)
    R_phi = (cfg.k_phi * (cfg.phi_bar - phi)- cfg.gamma * cfg.k_int * h_bet(phi) * max_val_psiP_0_h)
    
    return R_f, R_a, R_p, R_phi

# =============================================================================
# 3. VARIATIONAL FORMS
# =============================================================================
def build_variational_forms(psiF, zetaF, psiA, zetaA, psiP, zetaP, phi, zetaPhi, 
        muP, omgP, muPhi, omgPhi,psiF_n, psiA_n, psiP_n, phi_n, psiF_h, psiA_h, 
        psiP_h, phi_h, psiF_m, psiA_m, psiP_m, R_f, R_a, R_p, R_phi):

    rhoR_m  = rhoR(psiF_m, psiA_m, psiP_m)  # Picard iterate
    thetaVac_m = Theta_lambda_n(1 - rhoR_m)  # Regularised vacancy at Picard iterate

    # =============================================================================
    # VARIATIONAL FORMS
    # =============================================================================
    # ---------- psiF equation (Picard) ----------
    # LHS and RHS just for convenience. Does not matter really
    psi_f_LHS = ((psiF * zetaF) + (cfg.dt * delt(phi_n) * (thetaVac_m + psiF_m) * inner(grad(psiF), grad(zetaF))) + (cfg.dt * delt(phi_n) * psiF * inner(grad(psiA_m + psiP_m), grad(zetaF)))) * dx
    psi_f_RHS = (psiF_n * zetaF + cfg.dt * R_f * zetaF) * dx

    # ---------- psiA equation (Picard) ----------
    # LHS and RHS just for convenience. Does not matter really
    psi_a_LHS = ((psiA * zetaA) + (cfg.dt * delt(phi_n) * (thetaVac_m + psiA_m) * inner(grad(psiA), grad(zetaA))) + (cfg.dt * delt(phi_n) * psiA * inner(grad(psiF_m + psiP_m), grad(zetaA))) - (cfg.dt * A_a(phi_n) * psiA * thetaVac_m * inner(grad(phi_n), grad(zetaA))) ) * dx
    psi_a_RHS = (psiA_n * zetaA + cfg.dt * R_a * zetaA) * dx

    # ---------- psiP block (Schur) ----------
    # PAY ATTENTION to how this is written as the solver is designed for this specific form!
    # The logarithmic chemical potential ln(psiP/(1-rhoR)) has been absorbed analytically into this equation
    # could have been kept in mu_grdp, but wanted to avoid regularisaton/risk of blow up
    psiP_model = ((psiP * zetaP) - (psiP_n * zetaP + cfg.dt * R_p * zetaP) + (cfg.dt * delt(phi_n) * exp(-psiP_m/cfg.psiP_star) * (thetaVac_m + psiP_m) * inner(grad(psiP), grad(zetaP))) + (cfg.dt * delt(phi_n) * exp(-psiP_m/cfg.psiP_star) * psiP * inner(grad(psiF_m + psiA_m), grad(zetaP)))) * dx
    psiP_grdMu = (cfg.dt * max_value(M_p(psiP_m, rhoR_m, phi_n), as_ufl(1e-15)) * inner(grad(muP), grad(zetaP))) * dx
    mu_muP = (muP * omgP) * dx
    # The logarithmic term ln(psiP/(1-rhoR)) has been moved into psiP_model above.
    mu_grdp = ((cfg.N_p * phi_n * omgP) + (cfg.X_p * psiP * omgP) - (cfg.Lambda * inner(grad(psiP), grad(omgP))))* dx 
    
    # ---------- phi block (Schur) ----------
    # PAY ATTENTION to how this is written as the solver is designed for this specific form!
    phi_model = ((phi * zetaPhi) - (phi_n * zetaPhi + cfg.dt * R_phi * zetaPhi) ) * dx
    phi_grdMu = (cfg.dt * max_value(M_phi(phi_n), as_ufl(1e-10))  * inner(grad(muPhi), grad(zetaPhi))) * dx
    mu_muPhi  = (muPhi * omgPhi) * dx
    mu_grdphi = ((- cfg.K_eps * inner(grad(phi), grad(omgPhi))) - (cfg.K_phi * (dWc(phi) - dWe(phi_n)) * omgPhi) + (cfg.Pi_p_val * psiP_h * omgPhi) + (cfg.Pi_a_val * psiA_h * omgPhi)) * dx

    return (psi_f_LHS, psi_f_RHS, psi_a_LHS, psi_a_RHS, psiP_model, psiP_grdMu, mu_muP, mu_grdp, phi_model, phi_grdMu, mu_muPhi, mu_grdphi)

# =====================================================================================
# 4. OPERATORS AND SCHEMES
# =====================================================================================
def build_operators_and_schemes(psi_f_LHS, psi_f_RHS, psi_a_LHS, psi_a_RHS, psiP_model, 
    psiP_grdMu, mu_muP, mu_grdp,phi_model, phi_grdMu, mu_muPhi, mu_grdphi, psiP_h, phi_h):

    # --- psiF, psiA Schemes ---
    solver_param = {"nonlinear.tolerance": 1e-8, "nonlinear.verbose": False, "nonlinear.linesearch": "simple", "nonlinear.forcing": "eisenstatwalker",
    "linear.tolerance": 1e-8, "linear.verbose": False, "linear.preconditioning.method": "ssor", "linear.preconditioning.iteration": 2, "preconditioning.relaxation": 1.3}

    psi_f_Op = galerkinScheme(psi_f_LHS == psi_f_RHS, solver="gmres", parameters=solver_param)
    psi_a_Op = galerkinScheme(psi_a_LHS == psi_a_RHS, solver="gmres", parameters=solver_param)

    # --- psiP block: for SolverMPI ---
    # Creat a "nonlinear" operator from the provided weak form
    psiP_model_Op = galerkinOperator(psiP_model)  # A_op
    # Linearise the operator (around 0 by default), i.e. returns the assembled matrix
    Mat_psiP_model_Op = psiP_model_Op.linear()   # A_mat

    psiP_grdMu_Op = galerkinOperator(psiP_grdMu) # BT_op
    Mat_psiP_grdMu_Op = psiP_grdMu_Op.linear() # BT_mat

    mu_grdp_Op = galerkinOperator(mu_grdp) # B_op    
    Mat_mu_grdp_Op = mu_grdp_Op.linear() # B_mat

    mu_muP_Op = galerkinOperator(mu_muP)
    Mat_mu_muP_Op = mu_muP_Op.linear() # C_mat

    # --- phi block: for SolverMPI ---
    phi_model_Op = galerkinOperator(phi_model)  # A_op
    Mat_phi_model_Op = phi_model_Op.linear()   # A_mat

    phi_grdMu_Op = galerkinOperator(phi_grdMu) # BT_op
    Mat_phi_grdMu_Op = phi_grdMu_Op.linear() # BT_mat

    mu_grdphi_Op = galerkinOperator(mu_grdphi) # B_op
    # Create the matrix structure without assembling yet.
    Mat_mu_grdphi_Op = mu_grdphi_Op.linear(assemble=False)  # B_mat
    #mu_grdphi nonlinear via W'. Matrix (will be evaluated at phi_h) so depends on the current value of phi and needs to be reassembled at each Newton step.
    # Other matrices are constant and only need assembling once.
    mu_grdphi_Op.jacobian(phi_h, Mat_mu_grdphi_Op)

    mu_muPhi_Op  = galerkinOperator(mu_muPhi)
    Mat_mu_muPhi_Op = mu_muPhi_Op.linear() # C_mat

    # After AMR, jacobian() is still called to resize the matrix to the new DOF layout.

    ## ===> NOTE: BT_mat IS NOT NECESSARILY the transpose of B_mat. Just notation (bad one!) <===

    return (psi_f_Op, psi_a_Op, psiP_model_Op, Mat_psiP_model_Op, psiP_grdMu_Op, Mat_psiP_grdMu_Op, 
            mu_grdp_Op, Mat_mu_grdp_Op, mu_muP_Op, Mat_mu_muP_Op, phi_model_Op, Mat_phi_model_Op, 
            phi_grdMu_Op, Mat_phi_grdMu_Op, mu_grdphi_Op, Mat_mu_grdphi_Op, mu_muPhi_Op, Mat_mu_muPhi_Op)

# =============================================================================
# 5. SCHUR SOLVERS
# =============================================================================
def build_schur_solvers(psiP_model_Op, Mat_psiP_model_Op, psiP_grdMu_Op, Mat_psiP_grdMu_Op,
    mu_grdp_Op, Mat_mu_grdp_Op, Mat_mu_muP_Op, phi_model_Op, Mat_phi_model_Op,
    phi_grdMu_Op, Mat_phi_grdMu_Op, mu_grdphi_Op, Mat_mu_grdphi_Op, Mat_mu_muPhi_Op):

    # =============================================================================
    # SCHUR SOLVERS
    # =============================================================================
    # --- psiP block ---
    schur_params_psiP = {
        "method": cfg.linear_method,       # Linear solver: "fgmres" | "gmres" | "bcgs" | "cg"
        "tolerance": cfg.newtonTol,        # Newton tolerance
        "maxiterations": cfg.newtonMaxIt,     # Max Newton iterations
        "max_ksp_iters": cfg.linearMaxIt,    # Max linear solver iterations per Newton step
        "verbose": cfg.solver_verbose,
        "gmres_restart": cfg.gmresRestart,   # number of Krylov vectors stored before discarding subspace and restarting from current iterate
        "preconditioner": cfg.preconditioner,  # Preconditioner for linear solver: "bjacobi" | "hypre" | "jacobi" | "asm" | "none"
    }

    # Set up the solver with the correct operators/matrices
    # PAY ATTENTION and check solverMPI docstring for notation! Assigning wrong operators gives wrong results
    schur_solver_psiP = SchurSolver(A_op=psiP_model_Op, A_mat=Mat_psiP_model_Op, BT_op=psiP_grdMu_Op, BT_mat=Mat_psiP_grdMu_Op,
                                    B_op=mu_grdp_Op, B_mat=Mat_mu_grdp_Op, C_mat=Mat_mu_muP_Op, parameters=schur_params_psiP)

    # --- phi block ---
    schur_params_phi = {"method": cfg.linear_method, "tolerance": cfg.newtonTol, "maxiterations": cfg.newtonMaxIt, "max_ksp_iters": cfg.linearMaxIt,
                        "verbose": cfg.solver_verbose, "gmres_restart": cfg.gmresRestart, "preconditioner": cfg.preconditioner}

    schur_solver_phi = SchurSolver(A_op=phi_model_Op, A_mat=Mat_phi_model_Op, BT_op=phi_grdMu_Op, BT_mat=Mat_phi_grdMu_Op,
                                    B_op=mu_grdphi_Op, B_mat=Mat_mu_grdphi_Op, C_mat=Mat_mu_muPhi_Op, parameters=schur_params_phi)

    return schur_solver_psiP, schur_params_psiP, schur_solver_phi, schur_params_phi
