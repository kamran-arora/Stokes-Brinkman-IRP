"""Firedrake implementation of multigrid with Vanka smoother 
for the Taylor-Hood discretisation of the generalised Stokes problem:

    - Laplace u + u + grad(p) = f
                       div(u) = 0

See  https://dl.acm.org/doi/pdf/10.1145/3445791 for PatchPC smoothers       

"""

from timeit import default_timer as timer
from petsc4py import PETSc
from firedrake import *
from configurations import (
    ConfigurationSquareWithCylindricalHole,
    ConfigurationDualScaleDryFibres,
)
import matplotlib.pyplot as plt

# Use periodic BCs?
use_periodic_bcs = True

# Use multigrid?
use_multigrid = True

# Use direct solver?
use_direct_solver = False

# Compute reference solution?
compute_reference_solution = False

# smoother
smoother = "BraessSarazin"

# scaling parameter alpha in Braess-Sarazin preconditioner
alpha_bs = 1.25

# relative solver tolerance
ksp_rtol = 1.0e-6

# use PETSc options to configure Braess-Sarazin preconditioner?
use_petsc_options = True

# meshsize
nx = 64
ny = 64

# number of refinements, DEFAULT=3
nref = 3

print(f"Use periodic boundary conditions = {use_periodic_bcs}")
print(f"Use direct solver = {use_direct_solver}")
print(f"Use multigrid = {use_multigrid}")
print(f"smoother = {smoother}")
print(f"fine mesh = {nx} x {ny}")
print(f"number of refinements = {nref}")
print(f"KSP rtol = {ksp_rtol}")

# Select the configuration
config = ConfigurationSquareWithCylindricalHole(0.4)
# config = ConfigurationDualScaleDryFibres("regular")

if use_multigrid:
    if use_periodic_bcs:
        coarse_mesh = PeriodicUnitSquareMesh(
            nx // 2**nref, ny // 2**nref, quadrilateral=True
        )
    else:
        coarse_mesh = UnitSquareMesh(
            nx // 2**nref, ny // 2**nref, quadrilateral=True
        )
    mh = MeshHierarchy(coarse_mesh, refinement_levels=nref)
    mesh = mh[-1]
else:
    if use_periodic_bcs:
        mesh = PeriodicUnitSquareMesh(nx, ny)
    else:
        mesh = UnitSquareMesh(nx, ny)

# Taylor-Hood element
V_u = VectorFunctionSpace(mesh, "CG", 2)
V_p = FunctionSpace(mesh, "CG", 1)
V = V_u * V_p

u, p = TrialFunctions(V)
v, q = TestFunctions(V)

G = as_vector([1, 0])

f_rhs = Function(V_u).interpolate(G)

L = dot(v, f_rhs) * dx

beta = config.beta(mesh, ks=1e-6)

weak_form = (inner(grad(u), grad(v)) + beta * dot(u, v) + p * div(v) + q * div(u)) * dx


class PCScaledVelocityBlock(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        prefix = pc.getOptionsPrefix() + "scaled_velocity_block_"
        alpha = PETSc.Options().getReal(prefix + "alpha", 1.0)
        mesh = test.function_space().mesh()
        u, p = TestFunctions(test.function_space())
        v, q = TrialFunctions(trial.function_space())
        a = (
            alpha * (inner(grad(u), grad(v)) + config.beta(mesh, ks=1e-6) * dot(u, v))
            + p * div(v)
            + q * div(u)
        ) * dx
        A, _ = pc.getOperators()
        ctx = A.getPythonContext()
        return (a, ctx.bcs)


# Select the solver setup
if smoother == "Vanka":
    if use_multigrid:
        solver_parameters = {
            "ksp_type": "gmres",
            #"ksp_norm_type": "unpreconditioned",
            #"ksp_monitor_true_residual": None,
            "ksp_monitor": None,
            "ksp_rtol": ksp_rtol,
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
    else:
        solver_parameters = {
            "ksp_type": "gmres",
            "ksp_monitor": None,
            "ksp_rtol": ksp_rtol,
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
        }
elif smoother == "BraessSarazin":
    if use_multigrid:
        if use_petsc_options:
            solver_parameters = {
                "ksp_type": "gmres",
                #"ksp_monitor": None,
                "ksp_norm_type": "unpreconditioned",
                "ksp_monitor_true_residual": None,
                "ksp_rtol": ksp_rtol,
                "pc_type": "mg",
                "mat_type": "matfree",
                "mg_levels": {
                    "ksp_type": "richardson",
                    "ksp_max_it": 2,
                    "pc_type": "python",
                    "pc_python_type": __name__ + ".PCScaledVelocityBlock",
                    "scaled_velocity_block": {"alpha": alpha_bs},
                    "aux": {
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type": "schur",
                        "pc_fieldsplit_schur_factorization_type": "full",
                        "pc_fieldsplit_schur_precondition": "selfp",
                        "fieldsplit_0": {"ksp_type": "preonly", "pc_type": "jacobi"},
                        "fieldsplit_1": {
                            "ksp_type": "preonly",
                            "pc_type": "ksp",
                            "ksp_ksp_type": "gmres",
                            "ksp_pc_type": "hypre",
                            "ksp_ksp_rtol": ksp_rtol,
                        },
                    },
                },
                "mg_coarse": {
                    "ksp_type": "richardson",
                    "ksp_max_it": 5,
                    "pc_type": "python",
                    "pc_python_type": __name__ + ".PCScaledVelocityBlock",
                    "scaled_velocity_block": {"alpha": alpha_bs},
                    "aux": {
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type": "schur",
                        "pc_fieldsplit_schur_factorization_type": "full",
                        "pc_fieldsplit_schur_precondition": "selfp",
                        "fieldsplit_0": {"ksp_type": "preonly", "pc_type": "jacobi"},
                        "fieldsplit_1": {
                            "ksp_type": "preonly",
                            "pc_type": "ksp",
                            "ksp_ksp_type": "gmres",
                            "ksp_pc_type": "hypre",
                            "ksp_ksp_rtol": ksp_rtol,
                        },
                    },
                },
            }
        else:
            solver_parameters = {
                "ksp_type": "gmres",
                "ksp_monitor": None,
                #"ksp_norm_type": "unpreconditioned",
                #"ksp_monitor_true_residual": None,
                "ksp_rtol": ksp_rtol,
                "pc_type": "mg",
                "mat_type": "matfree",
                "mg_levels": {
                    "ksp_type": "richardson",
                    "ksp_max_it": 2,
                    "pc_type": "python",
                    "pc_python_type": "preconditioners.PCBraessSarazin",
                    "braess_sarazin": {
                        "alpha": alpha_bs,
                        "unit_diagonal": False,
                        "ksp_type": "gmres",
                    },
                },
                "mg_coarse": {
                    "ksp_type": "richardson",
                    "ksp_max_it": 5,
                    "pc_type": "python",
                    "pc_python_type": "preconditioners.PCBraessSarazin",
                    "braess_sarazin": {
                        "alpha": alpha_bs,
                        "unit_diagonal": False,
                        "ksp_type": "gmres",
                    },
                },
            }
    else:
        if use_petsc_options:
            solver_parameters = {
                "ksp_type": "gmres",
                "ksp_monitor": None,
                "ksp_rtol": ksp_rtol,
                "pc_type": "python",
                "mat_type": "matfree",
                "pc_python_type": __name__ + ".PCScaledVelocityBlock",
                "scaled_velocity_block": {"alpha": alpha_bs},
                "aux": {
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_factorization_type": "full",
                    "pc_fieldsplit_schur_precondition": "selfp",
                    "fieldsplit_0": {"ksp_type": "preonly", "pc_type": "jacobi"},
                    "fieldsplit_1": {
                        "ksp_type": "preonly",
                        "pc_type": "ksp",
                        "ksp_ksp_type": "cg",
                        "ksp_pc_type": "hypre",
                        "ksp_ksp_rtol": ksp_rtol,
                    },
                },
            }
        else:
            solver_parameters = {
                #"ksp_view": None,
                "ksp_type": "gmres",
                #"ksp_norm_type": "unpreconditioned",
                #"ksp_monitor_true_residual": None,
                "ksp_monitor": None,
                "ksp_rtol": ksp_rtol,
                "mat_type": "matfree",
                "pc_type": "python",
                "pc_python_type": "preconditioners.PCBraessSarazin",
                "braess_sarazin": {
                    "alpha": alpha_bs,
                    "unit_diagonal": False,
                    "ksp_type": "cg",
                    "pc_type": "hypre",
                    "ksp_rtol": ksp_rtol,
                },
            }
else:
    raise RuntimeError(f"Unknown smoother: {smoother}")

if use_direct_solver:
    solver_parameters = {
        "ksp_type": "gmres",
        "ksp_monitor": None,
        "ksp_rtol": ksp_rtol,
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

# Boundary conditions
if use_periodic_bcs:
    bcs = None
else:
    bcs = [DirichletBC(V.sub(0), as_vector([0, 0]), 1 + idx) for idx in range(4)]

# Nullspace: we can add an arbitrary constant to pressure
nullspace = MixedVectorSpaceBasis(
    V, [V.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)]
)

up = Function(V)

problem = LinearVariationalProblem(
    weak_form,
    L,
    up,
    bcs=bcs,
)

# Set solver
solver = LinearVariationalSolver(
    problem, solver_parameters=solver_parameters, nullspace=nullspace
)

t_start = timer()
solver.solve()
t_finish = timer()

t_elapsed = t_finish - t_start
print(f"elapsed time = {t_elapsed:6.2f} s")
print()
u_h, p_h = up.subfunctions
ndof_velocity = len(u_h.dat.data)
ndof_pressure = len(p_h.dat.data)

# subtract average pressure
p_h = assemble(p_h - assemble(p_h * dx))
u_h.rename("velocity")
p_h.rename("pressure")
fields = [u_h, p_h]

U = assemble(dot(u_h, G) * dx)
print("K/(Lx*Ly) = ", U / (config.Lx * config.Ly))
print()

print(f"number of velocity unknowns  = {ndof_velocity}")
print(f"number of pressure unknowns  = {ndof_pressure}")
print(f"---------------------------")
print(f"total number of unknowns     = {ndof_pressure+ndof_velocity}")
print()

# Compute "exact" reference solution, if requested
if compute_reference_solution:
    up_exact = Function(V)

    problem = LinearVariationalProblem(
        weak_form,
        L,
        up_exact,
        bcs=bcs,
    )
    exact_solver = LinearVariationalSolver(
        problem,
        solver_parameters={
            "ksp_type": "gmres",
            "ksp_monitor": None,
            "ksp_norm_type": "unpreconditioned",
            "ksp_monitor_true_residual": None,
            "ksp_rtol": 1e-6,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        nullspace=nullspace,
    )
    exact_solver.solve()
    u_h_exact, p_h_exact = up_exact.subfunctions

    # subtract average pressure
    p_h_exact = assemble(p_h_exact - assemble(p_h_exact * dx))
    u_h_exact.rename("velocity [exact]")
    p_h_exact.rename("pressure [exact]")
    # compute error
    p_h_error = assemble(p_h - p_h_exact)
    u_h_error = assemble(u_h - u_h_exact)
    p_h_error.rename("pressure [error]")
    u_h_error.rename("velocity [error]")
    fields += [u_h_exact, p_h_exact, p_h_error, u_h_error]

# Save all fields to disk
File("generalised_stokes.pvd").write(*fields)

from firedrake.pyplot import trisurf
surf = trisurf(u_h)
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(surf)
plt.tight_layout()
plt.show()