from firedrake import PCBase, assemble
from firedrake.matrix_free.operators import ImplicitMatrixContext
from petsc4py import PETSc


class PCBraessSarazin(PCBase):
    needs_python_pmat = True

    """Braess-Sarazin preconditioner

    Given the saddle point matrix

        [ A   B^T ]
        [         ]
        [ B    0  ]

    construct a preconditioner by replacing A by alpha * D where D is either the
    diagonal of A or the identity I.

    The parameters are controlled by two options:

        alpha: the value of the scaling factor
        unit_diagonal: use the identity matrix instead of the diagonal of A

    Solves the system

        [ alpha D   B^T ] [ u ]   [ r_u ]
        [               ] [   ] = [     ]
        [ B          0  ] [ p ] = [ r_p ]

    in three steps:

    1. compute modified right hand side

            tilde(r)_p = r_p - B (alpha D)^{-1} r_u

    2. solve the Schur-complement system

            S p = tilde(r)_p with S = - B (alpha D)^{-1} B^T

    3. back-substitute to obtain

            u = (alpha D)^{-1} (r_u - B^T p)

    """

    def initialize(self, pc):
        """Initialise preconditioner

        :arg pc: a Preconditioner instance.
        """
        prefix = pc.getOptionsPrefix() + "braess_sarazin_"
        _, P = pc.getOperators()
        self.ctx = P.getPythonContext()
        Pmat = assemble(self.ctx.a).petscmat
        if not isinstance(self.ctx, ImplicitMatrixContext):
            raise ValueError("The python context must be an ImplicitMatrixContext")
        # extract parameters
        self._alpha = PETSc.Options().getReal(prefix + "alpha", 1.0)
        self._unit_diagonal = PETSc.Options().getBool(prefix + "unit_diagonal", False)
        test, _ = self.ctx.a.arguments()
        V = test.function_space()
        # extract index sets
        self._ises = V._ises
        assert len(self._ises) == 2, "Function space must consist of two blocks"
        # extract matrix A from (0,0) block
        self._A = Pmat.createSubMatrix(self._ises[0], self._ises[0])
        # extract matrix B from (1,0) block
        self._B = Pmat.createSubMatrix(self._ises[1], self._ises[0])
        # construct - B^T from (0,1) block
        self._minus_BT = Pmat.createSubMatrix(self._ises[0], self._ises[1])
        self._minus_BT.scale(-1.0)
        # Construct diagonal matrix which is either (alpha*A)^{-1} or alpha^{-1} I
        diagonal = self._A.getDiagonal()
        if self._unit_diagonal:
            with diagonal as v:
                v[:] = 1.0 / self._alpha
        else:
            with diagonal as v:
                v[:] = 1.0 / (self._alpha * v[:])
        self._inv_diag = PETSc.Mat().createDiagonal(diagonal).convert("aij")
        # Schur complement S = - B (alpha D)^{-1} B^T
        Smat = self._B.matMult(self._inv_diag.matMult(self._minus_BT))
        schur_ksp = PETSc.KSP().create(comm=pc.comm)
        schur_ksp.incrementTabLevel(1, parent=pc)
        schur_ksp.setOptionsPrefix(prefix)
        schur_ksp.setOperators(Smat, Smat)
        schur_ksp.setFromOptions()
        self._schur_ksp = schur_ksp

    def update(self, pc):
        """Update preconditioner

        :arg pc: a Preconditioner instance."""
        pass

    def apply(self, pc, x, y):
        """Applies the Braess-Sarazin preconditioner.

        :arg pc: a Preconditioner instance.
        :arg x: A PETSc vector containing the incoming right-hand side.
        :arg y: A PETSc vector for the result.
        """
        # extract RHS (r_u, r_p)
        r_u = x.getSubVector(self._ises[0])
        r_p = x.getSubVector(self._ises[1])
        # extract solution
        u = y.getSubVector(self._ises[0])
        p = y.getSubVector(self._ises[1])
        # compute modified RHS tilde(r)_p = r_p - B (alpha D)^{-1} r_u for pressure
        inv_diag_r_u = r_u.duplicate()
        r_p_tilde = r_p.duplicate()
        self._inv_diag.mult(r_u, inv_diag_r_u)
        inv_diag_r_u.scale(-1.0)
        self._B.multAdd(inv_diag_r_u, r_p, r_p_tilde)
        # Solve for p
        self._schur_ksp.solve(r_p_tilde, p)
        # Reconstruct u = (alpha D)^{-1} (r_u - B^T p)
        u_tilde = u.duplicate()
        self._minus_BT.multAdd(p, r_u, u_tilde)
        self._inv_diag.mult(u_tilde, u)
        y.restoreSubVector(self._ises[0], u)
        y.restoreSubVector(self._ises[1], p)

    def applyTranspose(self, pc, x, y):
        """Apply the transpose of the preconditioner

        :arg pc: a Preconditioner instance.
        :arg x: A PETSc vector containing the incoming right-hand side.
        :arg y: A PETSc vector for the result.

        """

        raise NotImplementedError("Transpose application is not implemented.")

    def view(self, pc, viewer=None):
        """Viewer for Schur-complement KSP."""
        super().view(pc, viewer)
        viewer.printfASCII(f"alpha = {self._alpha}\n")
        viewer.printfASCII(f"unit_diagonal = {self._unit_diagonal}\n")
        viewer.printfASCII(f"KSP solver for Schur-complement:\n")
        self._schur_ksp.view(viewer)
