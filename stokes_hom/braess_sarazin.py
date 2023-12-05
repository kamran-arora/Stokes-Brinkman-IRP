from firedrake import *
import timeit

'''
Braess-Sarazin smoother
'''

def build_problem(mesh_size, parameters):
    # mesh
    N = 16
    mesh = PeriodicUnitSquareMesh(2**mesh_size, 2**mesh_size, quadrilateral=True)

    # function spaces
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    X = V * Q

    # functions
    u, p = TrialFunctions(X)
    v, q = TestFunctions(X)

    # bcs = [DirichletBC(X.sub(0), Constant((1, 0)), (4,)), DirichletBC(X.sub(0), Constant((0, 0)), (1, 2, 3))]

    a = (inner(grad(u), grad(v)) + inner(p, div(v)) - inner(q, div(u))) * dx
    L = inner(Constant((0, 0)), v) * dx

    nullspace = MixedVectorSpaceBasis(X, [X.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

    x = Function(X)
    vpb = LinearVariationalProblem(a, L, x)
    solver = LinearVariationalSolver(vpb, solver_parameters=parameters, nullspace=nullspace)

    return solver, x

################################################################################
# DIRECT SOLVER
# parameters = {"ksp_type": "gmres",
#               "mat_type": "aij",
#               "pc_type": "lu",
#               "pc_factor_mat_solver_type": "mumps"}

# for n in range(2, 8):
#     solver, x = build_problem(mesh_size=n, parameters=parameters)
#     t = timeit.default_timer()
#     solver.solve()
#     print(f"Mesh cells: {x.function_space().mesh().num_cells()}")
#     print(f"Time elapsed: {timeit.default_timer()-t}")
#     print("----------------------------------------")

################################################################################
# BRAESS-SARAZIN

# mat_type = matfree ??
# ksp_type = richardson
#   -  this is because Richardson is
#   - x_k+1 = x_k + w*P(b-A*x_k)
#   - so BS is this with P = (aD B^T B 0)^-1
# pc_fieldsplit_type = schur
# 

parameters = {"ksp_type": "richardson",
              "ksp_rtol": 1e-8,
              "ksp_monitor_true_residual": None,
              "pc_type": "fieldsplit",
              "pc_fieldsplit_type": "schur",
              "pc_fieldsplit_schur_fact_type": "full",
              "fieldsplit_schur_precondition": "selfp",
              "fieldsplit_0_ksp_type": "preonly",
              "fieldsplit_0_pc_type": "jacobi",
              "fieldsplit_1_ksp_type": "preonly",
              "fieldsplit_1_pc_type": "lu"}

for n in range(2, 7):
    solver, x = build_problem(mesh_size=n, parameters=parameters)
    t = timeit.default_timer()
    solver.solve()
    print(f"Mesh cells: {x.function_space().mesh().num_cells()}")
    print(f"Time elapsed: {timeit.default_timer()-t}")
    print(f"Iterations: {solver.snes.ksp.getIterationNumber()}")
    print("----------------------------------------")