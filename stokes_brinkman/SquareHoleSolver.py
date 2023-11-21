from firedrake import *
import timeit

class SquareHoleSolver():
    def __init__(self, N, R, ks, timer):
        # parameters
        self.N = N
        self.R = R
        self.ks = ks
        # mesh
        self.mesh = PeriodicUnitSquareMesh(self.N, self.N, quadrilateral=True)
        self.x, self.y = SpatialCoordinate(self.mesh)
        # beta
        self.beta = conditional((self.x-0.5)*(self.x-0.5)+(self.y-0.5)*(self.y-0.5) <= self.R*self.R, Constant(0), Constant(1/self.ks))
        # function spaces
        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)
        self.X = self.V * self.Q
        self.timer=timer

        # trial and test functions
        self.u, self.p = TrialFunctions(self.X)
        self.v, self.q = TestFunctions(self.X)

        # bilinear form
        self.a = (inner(grad(self.u), grad(self.v)) - inner(self.p, div(self.v)) - inner(self.q, div(self.u)) + self.beta * inner(self.u, self.v)) * dx

        # rhs
        self.f = Function(self.V).interpolate(as_vector([1, 0]))
        self.L = inner(self.f, self.v) * dx

        # nullspace
        self.nullspace = MixedVectorSpaceBasis(self.X, [self.X.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

    def solver(self):
        up = Function(self.X)
        t1 = timeit.default_timer()
        solve(self.a == self.L, up, nullspace=self.nullspace, solver_parameters={"ksp_type": "gmres",
                         "ksp_monitor": None,
                         "mat_type": "aij",
                         "pc_type": "lu",
                         "pc_factor_mat_solver_type": "mumps"})
        t2 = timeit.default_timer()
        u, p = up.subfunctions
        u.rename("Velocity")
        p.rename("Pressure")
        if self.timer:
            print(f"Time elapsed: {t2-t1}")
        return u, p
    
    def macroPerm(self, u):
        return assemble(dot(u, as_vector([1, 0])) * dx)