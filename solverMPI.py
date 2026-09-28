import time
import numpy as np
from functools import partial
from petsc4py import PETSc  # type: ignore
from dune.common import comm   # type: ignore
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None


class SchurSolver:
    """
    Schur complement based Newton solver for Cahn-Hilliard-type systems.

    Weak form example:
    phi_model = (phi * v) * dx - (phi_n * v) * dx - (dt * R(phi) * v) * dx       for given R
    phi_grdMu = (dt * M(phi) * inner(grad(mu), grad(v))) * dx
    mu_mu = (mu * psi) * dx
    mu_grdphi = -(eps * inner(grad(phi), grad(psi))) * dx - ((1/eps) * dW(phi) * psi) * dx  + m(phi) + h(any other coupling)   for given m and h
    
    These define the nonlinear block system:
        A*phi + BT*mu = 0    (phi-row, linear)
        B(phi) + C*mu = 0    (mu-row, nonlinear in phi due to dW(phi))

    Block matrix mapping:
    A  <--- phi_model  (phi-row, phi-column)
    BT <--- phi_grdMu  (phi-row, mu-column)
    B  <--- mu_grdphi  (mu-row,  phi-column) -- could be nonlinear in phi
    C  <--- mu_mu  (mu-row,  mu-column) -- should be SPD mass matrix 
    
    ===> NOTE: BT_mat IS NOT NECESSARILY the transpose of B_mat. Just notation (bad one!) <===
    
    The nonlinear block system is solved with a Newton loop with backtracking line search. 
    At each Newton iteration, the linearised system for the updates (delta_phi, delta_mu) is:
        [ A    BT ] [ delta_phi ]   [ b_phi ]
        [ B     C ] [ delta_mu  ] = [ f     ]   
        
        The RHS is the negative residual of the current iterate (computed each Newton step):
        b_phi = -A*phi - BT*mu   (phi-row residual)
        f     = -B(phi) - C*mu     (mu-row residual)
    
    This linearised system is solved via the Schur complement:
    ---->    S * delta_phi = rhs_schur = b_phi - BT * C^{-1} * f
    where S = A - BT * C^{-1} * B is the Schur complement of C in the block system.
    S is never formed explicitly. It is applied matrix-free via a PETSc shell matrix.
    mu is then recomputed directly from the constitutive constraint C*t2 = -B(t1)
    at each line search trial point, keeping t2 consistent with the trial t1.
    
    Code Notation
    --------------
    target1= phi ---> t1  ||  target2= mu ---> t2
    
    Parameters
    ----------
    method: str -- linear solver (default: "fgmres")
    tolerance: float -- relative Newton convergence tolerance (default: 1e-5)
    absolute_tolerance: float -- absolute Newton residual tolerance (default: 1e-12)
    maxiterations: int -- max Newton iterations (default: 20)
    max_ksp_iters: int -- max linear solver (KSP) iterations (default: 300)
    gmres_restart: int -- GMRES restart size (default: 100)
    verbose: bool -- print Newton/KSP info (default: False)
    preconditioner: str -- preconditioner for S (default: "bjacobi")

    Available linear solvers (method parameter)
    -------------------------------------------
    "fgmres" : Flexible GMRES (default).  
    "gmres" : Standard GMRES.  
    "bcgs" : BiCGSTAB. 
    "cg" : Conjugate Gradient. 

    Available preconditioners (preconditioner parameter)
    ----------------------------------------------------
    "bjacobi" : Block Jacobi + ILU(0).
    "hypre" : Algebraic multigrid via HYPRE BoomerAMG. 
    "jacobi" : Diagonal scaling only.
    "asm" : Additive Schwarz Method.
    """

    def __init__(self, A_op, A_mat, BT_op, BT_mat, B_op, B_mat, C_mat, parameters=None):
        self.A_op = A_op
        self.A_mat = A_mat
        self.BT_mat = BT_mat
        self.B_op = B_op
        self.B_mat = B_mat
        self.C_mat = C_mat

        params = parameters or {}

        self.method = params.get("method", "fgmres")

        # Newton stopping tolerances:
        # 'tolerance' is interpreted as a RELATIVE tolerance.
        self.rtol = params.get("tolerance", 1e-5)
        self.atol = params.get("absolute_tolerance", 1e-12)

        self.maxiter = params.get("maxiterations", 20)
        self.max_ksp_iters = params.get("max_ksp_iters", 300)
        self.restart = params.get("gmres_restart", 100)
        self.verbose = params.get("verbose", False)
        self.preconditioner = params.get("preconditioner", "bjacobi")

        # Raise error if method provided is not valid
        _valid_methods = ("fgmres", "gmres", "bcgs", "cg")
        if self.method not in _valid_methods:
            raise ValueError(f"Method '{self.method}' not available. Available methods: {_valid_methods}")

        # Raise error if preconditioner provided is not valid
        _valid_preconditioners = ("bjacobi", "hypre", "jacobi", "asm", "none")
        if self.preconditioner not in _valid_preconditioners:
            raise ValueError(f"Preconditioner '{self.preconditioner}' not available. Available preconditioners: {_valid_preconditioners}")

        # Build all persistent objects
        self._build_persistent_objects()

    def _build_persistent_objects(self):
        """Create (or recreate) all persistent KSPs and pre-allocated vectors.
        Called once in __init__ and again after AMR via rebuild_after_amr()."""
        C_petsc = self.C_mat.as_petsc
        A_petsc = self.A_mat.as_petsc

        # Persistent KSP for C^{-1}
        self._ksp_Cinv = PETSc.KSP().create(PETSc.COMM_WORLD)  # Create solver across all MPI ranks
        self._ksp_Cinv.setType(PETSc.KSP.Type.CG)  # C is SPD mass matrix ---> CG is optimal
        self._ksp_Cinv.getPC().setType(PETSc.PC.Type.BJACOBI)  # Block Jacobi preconditioner
        self._ksp_Cinv.setOperators(C_petsc, C_petsc)  # First argument: matrix to solve with. Second: matrix to build preconditioner from
        self._ksp_Cinv.setTolerances(rtol=0.0, atol=1e-14, max_it=500) 
        self._ksp_Cinv.setInitialGuessNonzero(False)         
        self._ksp_Cinv.setUp()                                

        # Schur solve S * delta_phi = rhs_schur
        self._ksp_schur = PETSc.KSP().create(PETSc.COMM_WORLD)  # Create solver across all MPI ranks
        method_map = {"fgmres": PETSc.KSP.Type.FGMRES, "gmres": PETSc.KSP.Type.GMRES, "bcgs": PETSc.KSP.Type.BCGS, "cg": PETSc.KSP.Type.CG}
        self._ksp_schur.setType(method_map[self.method])  # Set solver type from method_map
        if self.method in ("fgmres", "gmres"):
            self._ksp_schur.setGMRESRestart(self.restart) 
        pc = self._ksp_schur.getPC()  # Setting up preconditioner
        if self.preconditioner == "hypre":
            pc.setType(PETSc.PC.Type.HYPRE)  # AMG via HYPRE BoomerAMG
            pc.setHYPREType("boomeramg")
        elif self.preconditioner == "jacobi":
            pc.setType(PETSc.PC.Type.JACOBI)    # Diagonal scaling
        elif self.preconditioner == "asm":
            pc.setType(PETSc.PC.Type.ASM)   # Additive Schwarz
        elif self.preconditioner == "none":
            pc.setType(PETSc.PC.Type.NONE)   # No preconditioning    
        else:
            pc.setType(PETSc.PC.Type.BJACOBI)      # ILU(A) per rank (default)
        self._ksp_schur.setInitialGuessNonzero(False)  

        # Pre-allocated vectors for solve()
        self._CinvF = C_petsc.createVecRight()   # Create a vector matching the column size of C_petsc for C^{-1} * f
        self._rhs_schur = A_petsc.createVecLeft()    # Schur RHS
        self._delta_t1 = A_petsc.createVecRight()   # Newton update for target1
        self._delta_t2 = C_petsc.createVecRight()   # Newton update for target2
        self._rhs_t2 = C_petsc.createVecLeft()    # f - B * delta_target1

        self._g = A_petsc.createVecLeft()   # target1-row residual
        self._f = C_petsc.createVecLeft()   # target2-row residual

        # Temporary vectors for Schur shell matvec
        self._tmp_shell_in = C_petsc.createVecLeft()  # Temporary vectors matching the row size of C_petsc (target2 DOFs)
        self._tmp_shell_out = C_petsc.createVecLeft()

        # Persistent shell matrix for S = A - BT*C^{-1}*B (matrix-free)
        n_local = A_petsc.getLocalSize()[0]   # Owned DOFs on this rank
        n_global = A_petsc.getSize()[0]  # Total DOFs across all ranks
        ctx = _SchurContext(A_petsc, self.BT_mat.as_petsc, self.B_mat.as_petsc,self._apply_Cinv, self._tmp_shell_in, self._tmp_shell_out)
        self._shell = PETSc.Mat().createPython(((n_local, n_global), (n_local, n_global)), context=ctx, comm=PETSc.COMM_WORLD)
        # context=ctx --->  attaches SchurContext so PETSc calls ctx.mult when it needs S*x
        # comm=PETSc.COMM_WORLD --->  matrix is distributed across all MPI ranks
        # (n_local, n_global) ---> tells PETSc the local (each rank) and global (all ranks) dimensions for rows and columns
        self._shell.setUp()  

        self._t1_old = A_petsc.createVecRight()

        self._workspace_t1 = None
        self._workspace_t2 = None

        # setOperators and setUp called once here, not inside the Newton loop
        # S_approx = A - BT*diag(C)^{-1}*B: explicit sparse approximation of the Schur complement
        # Used as PC matrix instead of A alone! Gives preconditioner correct scale and structure of S
        self._S_approx = self._assemble_S_approx()
        self._ksp_schur.setOperators(self._shell, self._S_approx)
        self._ksp_schur.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED) 
        self._ksp_schur.setUp()
        

    @staticmethod
    def _destroy_vec(v):
        """Safely destroy a PETSc vector if it exists."""
        if v is not None:
            try:
                v.destroy()
            except Exception:
                pass
    
    # All persistent objects will need to eb rebuild after adaptation
    def rebuild_after_amr(self):
        """Rebuild all persistent objects after mesh adaptation changes DOF layout.
        Must be called after jacobian reassembly calls following gridAdapt."""
        self._ksp_Cinv.destroy()   
        self._ksp_schur.destroy()
        self._shell.destroy()      
            
        # Destroy all pre-allocated vectors before rebuilding
        for v in [self._CinvF, self._rhs_schur, self._delta_t1, self._delta_t2, self._rhs_t2,
            self._g, self._f, self._tmp_shell_in, self._tmp_shell_out, self._t1_old]:
            
            self._destroy_vec(v)

        # Destroy S_approx before rebuilding
        if hasattr(self, '_S_approx') and self._S_approx is not None:
            self._S_approx.destroy()
            self._S_approx = None
        # Reset workspace DUNE functions 
        self._workspace_t1 = None
        self._workspace_t2 = None

        self._build_persistent_objects()

    # A function to apply C^{-1}
    def _apply_Cinv(self, vec_in_petsc, vec_out_petsc):
        """Solve C * vec_out = vec_in via persistent CG + BJACOBI KSP."""
        vec_out_petsc.set(0.0)
        self._ksp_Cinv.solve(vec_in_petsc, vec_out_petsc)

        reason = self._ksp_Cinv.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f"C-inverse solve failed: reason={reason}, "
                f"iterations={self._ksp_Cinv.getIterationNumber()}, "
                f"residual={self._ksp_Cinv.getResidualNorm():.3e}.")


    def _recompute_t2_from_constraint(self, target1, target2):
        """Recompute target2 (mu) directly from the constitutive constraint C*mu = -B(t1)."""
        self.B_op(target1, self._workspace_t2)    # workspace_t2 = B(t1)
        self._rhs_t2.array[:] = -self._workspace_t2.as_petsc.getArray()[:self._rhs_t2.local_size]  # rhs_t2 = -B(t1)
        self._rhs_t2.assemble()
        self._apply_Cinv(self._rhs_t2, self._delta_t2)  # delta_t2 = C^{-1} * (-B(t1))
        target2.as_petsc.getArray()[:self._delta_t2.local_size] = self._delta_t2.getArray()  # Write into target2
        target2.as_petsc.assemble()
    
    def recompute_target2_from_constraint(self, target1, target2):
        """Recompute target2 so that C*target2 = -B(target1)."""
        if self._workspace_t1 is None:
            self._workspace_t1 = target1.copy()
            self._workspace_t2 = target2.copy()

        self._recompute_t2_from_constraint(target1, target2)
        _ghost_update(target2.as_petsc)
    
    def _assemble_S_approx(self):
        """Assemble S_approx = A - BT * diag(C)^{-1} * B as an explicit sparse matrix.
        Used as the PC matrix for the Schur KSP instead of A alone.
        diag(C)^{-1} is a lumped (diagonal) approximation of C^{-1}."""
        # Lumped C^{-1}: diagonal approximation of C^{-1}
        diag_C = self.C_mat.as_petsc.getDiagonal()
        diag_C.reciprocal()  # diag_C = 1/diag(C)

        # CinvB = diag(C)^{-1} * B  (left-scale rows of B by diag_C)
        CinvB = self.B_mat.as_petsc.copy()
        CinvB.diagonalScale(L=diag_C)

        # BTCinvB = BT * CinvB
        BTCinvB = self.BT_mat.as_petsc.matMult(CinvB)

        # S_approx = A - BT * diag(C)^{-1} * B
        S_approx = self.A_mat.as_petsc.copy()
        S_approx.axpy(-1.0, BTCinvB)
        S_approx.assemble()

        # Clean up temporaries
        CinvB.destroy()
        BTCinvB.destroy()
        diag_C.destroy()
        return S_approx

    def _extract_reduced_residual(self, target1, workspace_t1, workspace_t2):
        """Compute the reduced nonlinear residual
        R_red(target1) = A(target1) - BT * C^{-1} * B(target1).
        """

        # A(target1)
        self.A_op(target1, workspace_t1)
        self._rhs_schur.array[:] = workspace_t1.as_petsc.getArray()[:self._rhs_schur.local_size]
        self._rhs_schur.assemble()

        # B(target1)
        self.B_op(target1, workspace_t2)
        self._f.array[:] = workspace_t2.as_petsc.getArray()[:self._f.local_size]
        self._f.assemble()

        # C^{-1} B(target1)
        self._apply_Cinv(self._f, self._CinvF)

        # BT C^{-1} B(target1)
        self._g.set(0.0)
        self.BT_mat.as_petsc.mult(self._CinvF, self._g)

        # R_red = A(target1) - BT C^{-1} B(target1)
        self._rhs_schur.axpy(-1.0, self._g)
        self._rhs_schur.assemble()

        return self._rhs_schur
    def solve(self, target1, target2):
        """Newton solver. Updates target1 and target2.
        target2 is recomputed from the constitutive constraint C*t2 = -B(t1)."""
        
        t_start = time.time()

        # Reuse persistent workspace DUNE functions
        if self._workspace_t1 is None:
            self._workspace_t1 = target1.copy()
            self._workspace_t2 = target2.copy()
        workspace_t1 = self._workspace_t1
        workspace_t2 = self._workspace_t2
        
        # Calculate initial reduced residual
        red_residual = self._extract_reduced_residual(target1, workspace_t1, workspace_t2)
        initial_residual = float(np.sqrt(red_residual.dot(red_residual)))
        # Newton stopping threshold
        # self.rtol is the user relative tolerance, e.g. 1e-5.
        # self.atol is only a small absolute safeguard, e.g. 1e-12.
        newton_threshold = max(self.atol, self.rtol * initial_residual)

        if self.verbose:
            print(
                f"    >>> NEWTON SOLVE | initial ||r|| = {initial_residual:.4e} | rtol={self.rtol:.1e} | atol={self.atol:.1e} | threshold={newton_threshold:.4e}",flush=True)
            
        total_lin_iters = 0
        newton_iter = 0
        newton_converged = False
        final_residual = initial_residual
        old_residual = initial_residual
        rhs_norm = 0.0
        n_iters = 0
        reason = 0
        final_ksp_residual = 0.0
        _exit_reason = "maxiter reached"  # default if while condition expires naturally
        
        while (not newton_converged) and (newton_iter < self.maxiter):
            newton_iter += 1
            n_iters = 0  # reset each iteration
            if self.verbose:
                print(f"      --- Newton iter {newton_iter} ---", flush=True)


            # Reassemble Jacobian blocks at the current Newton iterate
            self.A_op.jacobian(target1, self.A_mat)
            self.B_op.jacobian(target1, self.B_mat)

            if self.verbose:
                print("      A_mat and B_mat reassembled at current Newton iterate", flush=True)

            # Rebuild Schur preconditioner so it matches the current Jacobian blocks
            self._S_approx.destroy()
            self._S_approx = self._assemble_S_approx()
            self._ksp_schur.setOperators(self._shell, self._S_approx)

            # Compute reduced nonlinear residual
            red_residual = self._extract_reduced_residual(target1, workspace_t1, workspace_t2)
            current_residual = float(np.sqrt(red_residual.dot(red_residual)))

            # Newton right-hand side: J_red * delta_t1 = -R_red
            self._rhs_schur.scale(-1.0)
            rhs_norm = current_residual
            
            # --- Diagnostic [DIAG] prints ---
            if self.verbose:
                print(f"      [DIAG] ||R_red||={current_residual:.4e}", flush=True)
            
            # Reduced Newton convergence check.
            if current_residual <= newton_threshold:
                final_residual = current_residual
                newton_converged = True
                break

            # Start from zero IC
            self._delta_t1.set(0.0)
            self._ksp_schur.setInitialGuessNonzero(False)

            # Solve J_red * delta_t1 = -R_red
            linear_rtol = 1e-4
            linear_atol = 1e-14

            self._ksp_schur.setTolerances(rtol=linear_rtol, atol=linear_atol, max_it=self.max_ksp_iters)
            self._ksp_schur.solve(self._rhs_schur, self._delta_t1)

            n_iters = self._ksp_schur.getIterationNumber()
            reason = self._ksp_schur.getConvergedReason()
            final_ksp_residual = self._ksp_schur.getResidualNorm()

            # Compute true Schur residual to verify KSP accuracy
            if self.verbose:
                _Sd1 = self._rhs_schur.duplicate()
                self._shell.mult(self._delta_t1, _Sd1)
                _Sd1.axpy(-1.0, self._rhs_schur)
                schur_abs = float(np.sqrt(_Sd1.dot(_Sd1)))
                schur_rel = schur_abs / max(rhs_norm, 1e-30)
                _Sd1.destroy()
                print(f"        [KSP CHECK] true ||S*delta_t1-rhs||={schur_abs:.4e} | rel={schur_rel:.4e}", flush=True)

            if reason == 2:
                tol_used = f"rtol threshold={linear_rtol * rhs_norm:.4e}"
            elif reason == 3:
                tol_used = f"atol={linear_atol:.4e}"
            else:
                tol_used = (f"neither rtol={linear_rtol * rhs_norm:.4e} nor atol={linear_atol:.4e} satisfied")

            if self.verbose:
                print(f"        KSP Converged: reason={reason}, iters={n_iters}, residual={final_ksp_residual:.4e} ({tol_used})", flush=True)
            if reason < 0:
                print(f"        WARNING -- KSP did not converge: reason={reason}, iters={n_iters}, residual={final_ksp_residual:.4e} ({tol_used})", flush=True)

            total_lin_iters += n_iters

            if self.verbose:
                _norm_d1_diag = float(np.sqrt(self._delta_t1.dot(self._delta_t1)))
                print(f"      [DIAG] ||delta_t1||={_norm_d1_diag:.4e}", flush=True)

            # -----------------------------------------------------------------------
            # Line search on the reduced residual R_red(t1)
            # -----------------------------------------------------------------------
            target1.as_petsc.copy(result=self._t1_old)  # save t1 for restore on rejection
            old_residual = current_residual

            alpha = 1.0
            accepted = False
            ls_iters = 0
            max_damp_attempts = 10
            armijo_c = 1e-6

            while ls_iters < max_damp_attempts:
                # Restore t1 to pre-line-search state and apply trial step
                self._t1_old.copy(result=target1.as_petsc)
                target1.as_petsc.axpy(alpha, self._delta_t1)
                target1.as_petsc.assemble()
                _ghost_update(target1.as_petsc)

                # Compute reduced residual at trial t1
                red_residual_trial = self._extract_reduced_residual(target1, workspace_t1, workspace_t2)
                trial_residual = float(np.sqrt(red_residual_trial.dot(red_residual_trial)))

                if np.isfinite(trial_residual) and (trial_residual <= old_residual * (1.0 - armijo_c * alpha) or trial_residual <= newton_threshold):
                    final_residual = trial_residual
                    accepted = True
                    if self.verbose:
                        print(f"        [LS] alpha={alpha:.3e} --> ||R_red||={trial_residual:.4e} (old={old_residual:.4e})", flush=True)
                    break

                if self.verbose:
                    print(f"        [LS reject] alpha={alpha:.3e} --> ||R_red||={trial_residual:.4e} (old={old_residual:.4e}, need<={old_residual * (1.0 - armijo_c * alpha):.4e})", flush=True)

                alpha *= 0.5
                ls_iters += 1

            if not accepted:
                # Restore t1 to pre-line-search state
                self._t1_old.copy(result=target1.as_petsc)
                target1.as_petsc.assemble()
                _ghost_update(target1.as_petsc)

                if self.verbose:
                    print("      Line search failed -- stopping", flush=True)

                # Residual of the restored reduced state
                red_residual = self._extract_reduced_residual(target1, workspace_t1, workspace_t2)
                final_residual = float(np.sqrt(red_residual.dot(red_residual)))

                _exit_reason = f"Line search failed (all {max_damp_attempts} attempts)"
                break

            newton_converged = final_residual <= newton_threshold

        # Final convergence check
        newton_converged = final_residual <= newton_threshold

        # Diagnostics on failure
        if not newton_converged:
            red_residual = self._extract_reduced_residual(target1, workspace_t1, workspace_t2)
            _norm_red = float(np.sqrt(red_residual.dot(red_residual)))
            _norm_d1 = float(np.sqrt(self._delta_t1.dot(self._delta_t1)))
            _norm_A  = self.A_mat.as_petsc.norm()
            _norm_BT = self.BT_mat.as_petsc.norm()
            _norm_B  = self.B_mat.as_petsc.norm()
            _norm_C  = self.C_mat.as_petsc.norm()

            print(f"        [DIAG] ||R_red|| = {_norm_red:.4e}", flush=True)
            print(f"        [DIAG] ||delta_t1|| = {_norm_d1:.4e}", flush=True)
            print(f"        [DIAG] ||A|| = {_norm_A:.4e}", flush=True)
            print(f"        [DIAG] ||BT|| = {_norm_BT:.4e}", flush=True)
            print(f"        [DIAG] ||B|| = {_norm_B:.4e}", flush=True)
            print(f"        [DIAG] ||C|| = {_norm_C:.4e}", flush=True)
            print(f"        [DIAG] old_residual = {old_residual:.4e}", flush=True)
            print(f"        [DIAG] KSP reason = {reason}", flush=True)
            print(f"        [DIAG] KSP residual = {final_ksp_residual:.4e}", flush=True)

        if self.verbose and newton_converged:
            print(f"    <<< NEWTON CONVERGED in {newton_iter} iters | ||r||: {initial_residual:.4e} --> {final_residual:.4e} | threshold={newton_threshold:.4e} (rtol={self.rtol:.1e}, atol={self.atol:.1e})", flush=True)

        if not newton_converged:
            print(f"    WARNING: Newton did not converge after {newton_iter} iters (max={self.maxiter}) -- reason: {_exit_reason} | residual={final_residual:.4e} | threshold={newton_threshold:.4e} (rtol={self.rtol:.1e}, atol={self.atol:.1e})", flush=True)
        
        # Reconstruct target2 from the final target1
        if newton_converged:
            self._recompute_t2_from_constraint(target1, target2)

        _ghost_update(target1.as_petsc)
        _ghost_update(target2.as_petsc)

        t_end = time.time()
        return {
            "converged": newton_converged,
            "iterations": newton_iter,
            "linear_iters": total_lin_iters,
            "time": t_end - t_start,
            "initial_residual": initial_residual,
            "residual": final_residual,
            "threshold": newton_threshold,
            "rtol": self.rtol,
            "atol": self.atol,
        }


# Update ghost DOFs held by each rank. ghost DOFs = read-only copies of neighbouring ranks' boundary DOFs stored locally
def _ghost_update(vec):
    try:  
        vec.ghostUpdateBegin(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  
        vec.ghostUpdateEnd(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)    
    except PETSc.Error:
        pass
# INSERT: received value replaces the ghost copy. Alternative ---> ADD: add the received value to current ghost copy. Used when accumulating contributions
# FORWARD: communication goes in the direction: owner → ghost. Alternative ---> REVERSE: ghosts send their values back to the owning rank


# Define the matrix-free action of Schur S
class _SchurContext:
    """Shell matrix context for S = A - BT*C^{-1}*B."""

    def __init__(self, A_petsc, BT_petsc, B_petsc, apply_Cinv, tmp_in, tmp_out):
        self.A = A_petsc
        self.BT = BT_petsc
        self.B = B_petsc
        self.Cinv = apply_Cinv
        self.tmp_in = tmp_in
        self.tmp_out = tmp_out

    def mult(self, mat, x, y):
        """y = (A - BT*C^{-1}*B) * x"""
        self.B.mult(x, self.tmp_in)    # tmp_in = B * x
        self.Cinv(self.tmp_in, self.tmp_out)  # tmp_out = C^{-1} * B * x
        self.BT.mult(self.tmp_out, y)   # y = BT * C^{-1} * B * x
        y.scale(-1.0)   # y = -BT * C^{-1} * B * x
        self.A.multAdd(x, y, y)  # y += A * x
