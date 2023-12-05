from firedrake import *
import timeit
from periodicrectanglemeshhierarchy import PeriodicRectangleMeshHierarchy

def build_problem(mesh_size, parameters, R, ks):

    mesh = PeriodicUnitSquareMesh(2**mesh_size, 2**mesh_size, quadrilateral=True)

    x, y = SpatialCoordinate(mesh)
    beta = conditional((x - 0.5) ** 2 + (y - 0.5) ** 2 < R**2, 0.0, 1 / ks)

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    X = V * Q

    u, p = TrialFunctions(X)
    v, q = TestFunctions(X)

    a = (inner(grad(u), grad(v)) - inner(p, div(v)) - inner(q, div(u)) + beta * inner(u, v)) * dx

    f = Function(V).interpolate(as_vector([1, 0]))
    L = inner(f, v) * dx

    x = Function(X)

    nullspace = MixedVectorSpaceBasis(X, [X.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

    problem = LinearVariationalProblem(a, L, x)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters, nullspace=nullspace)

    return solver, x

################################################################################
# DIRECT SOLVER
parameters = {"ksp_type": "gmres",
              "mat_type": "aij",
              "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}

for n in range(2, 6):
    solver, x = build_problem(mesh_size=n, parameters=parameters, R=0.4, ks=1e-6)
    t1 = timeit.default_timer()
    solver.solve()
    t2 = timeit.default_timer()
    print(f"Mesh cells: {x.function_space().mesh().num_cells()}")
    print(f"Time elapsed: {t2 - t1}")
    print("----------------------------------------")

################################################################################
def build_problem_mg(mesh_size, parameters, R, ks, levels):

    mesh_hierarchy = PeriodicRectangleMeshHierarchy(2**mesh_size, 2**mesh_size, levels)

    mesh = mesh_hierarchy[-1]

    x, y = SpatialCoordinate(mesh)
    beta = conditional((x - 0.5) ** 2 + (y - 0.5) ** 2 < R**2, Constant(0), Constant(1 / ks))

    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    X = V * Q

    u, p = TrialFunctions(X)
    v, q = TestFunctions(X)

    a = (inner(grad(u), grad(v)) - inner(p, div(v)) - inner(q, div(u)) + beta * inner(u, v)) * dx

    f = Function(V).interpolate(as_vector([1, 0]))
    L = inner(f, v) * dx

    x = Function(X)

    nullspace = MixedVectorSpaceBasis(X, [X.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

    problem = LinearVariationalProblem(a, L, x)
    solver = LinearVariationalSolver(problem, solver_parameters=parameters, nullspace=nullspace)

    return solver, x

################################################################################
parameters = {
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "ksp_rtol": 1.0e-6,
    "pc_type": "mg",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        "patch": {
            "sub_ksp_type": "preonly",
            "sub_pc_type": "lu",
            "pc_patch": {
                "local_type": "additive",
                "partition_of_unity": False,
                "construct_type": "vanka",
                "construct_dim": 0,
                "exclude_subspaces": 1,
                "sub_mat_type": "seqdense",
                "save_operators": True,
            },
        },
    },
    "mg_coarse": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

solver, x = build_problem_mg(mesh_size=7, parameters=parameters, R=0.4, ks=1e-6, levels=1)
t1 = timeit.default_timer()
solver.solve()
t2 = timeit.default_timer()
print(f"Mesh cells: {x.function_space().mesh().num_cells()}")
print(f"Time elapsed: {t2 - t1}")
print("----------------------------------------")

################################################################################