from firedrake import *
from timeit import default_timer as timer
from petsc4py import PETSc
from configurations import ParallelChannel
from firedrake.__future__ import interpolate
import matplotlib.pyplot as plt
import os
import numpy as np


def parallel_channel_solver(nx, ny, ksp_rtol=1e-6, a=0.1, b=0.1, ks=1e-6, smoother='direct', mg=False, show=True):

    '''
    Solve 1D parallel channel problem with periodic boundary conditions on a 2D periodic rectangle mesh of size [0, a+b] x [0, 1].

    Parameters
    --------------------------
    nx: number of cells in x-direction
    ny: number of cells in y-direction
    ksp_rtol: relative tolerance of solver
    a: length of fluid region
    b: length of solid region
    ks: microscopic permeability
    solver: "direct", "vanka", "braess-sarazin"
    mg: toggle multigrid
    show: toggle plt.show()
    '''

    # scaling factor for braess-sarazin
    alpha_bs = 1.25

    # viscosity
    mu = 1
    # brinkman viscosity
    mue = 1

    # check for valid solver option
    if smoother not in ['direct', 'vanka', 'braess-sarazin']:
        raise Exception(f"Smoother must be one of 'direct', 'vanka', 'braess-sarazin' but you provided '{solver}'.")

    # mesh
    Lx = a + b
    if not mg:
        mesh = PeriodicRectangleMesh(nx=nx, ny=ny, Lx=Lx, Ly=1)
    elif mg:
        # nx should be a power of 2
        K = np.log2(nx)
        if mg and K < 4:
            raise Exception("For multigrid, nx and ny must be >= 16")
        nref = int(K-3)
        print(f"Number of refinements: {nref}")
        print(nx//2**nref)
        coarse_mesh = PeriodicRectangleMesh(nx=nx//2**nref, ny = ny//2**nref, Lx=Lx, Ly=1)
        mh = MeshHierarchy(coarse_mesh, refinement_levels=nref)
        mesh = mh[-1]

    # function space
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q

    # trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # pressure loading condition
    plc = 1.0
    #G = Constant(plc)
    f_rhs = Function(V).interpolate(as_vector([0, plc]))

    # rhs
    L = inner(v, f_rhs)*dx

    # beta
    config = ParallelChannel(a=a, b=b)
    beta = config.beta(mesh, ks=ks)

    # coordinates on mesh
    x, y = SpatialCoordinate(mesh)
    coords = Function(V).interpolate(as_vector([x, y]))
    x_coord = coords.sub(0)

    # weak form
    weak_form = (-inner(grad(u), grad(v)) - beta*inner(u, v) + p*div(v) + q*div(u))*dx

    # solver parameters
    if smoother=="vanka":
        if mg:
            solver_parameters = {
            "ksp_type": "gmres",
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
    elif smoother=="braess-sarazin":
        if mg:
            solver_parameters = {
                "ksp_type": "gmres",
                "ksp_monitor": None,
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
            solver_parameters = {
                "ksp_type": "gmres",
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

    # solver options
    if smoother=="direct":
        solver_parameters = {
            "ksp_type": "gmres",
            "ksp_monitor": None,
            "ksp_rtol": ksp_rtol,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }


    # solution
    up = Function(W)

    # boundary conditions
    bcs = None

    # define problem
    problem = LinearVariationalProblem(weak_form, L, up, bcs=bcs)

    # nullspace
    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True, comm=COMM_WORLD)])

    # define solver 
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullspace)

    # time the solver
    t_start = timer()
    solver.solve()
    t_finish = timer()

    # print time to solve
    t_elapsed = t_finish - t_start
    print(f"elapsed time = {t_elapsed:6.2f} s")
    print()

    # extract velocity in y-direction
    uu = up.sub(0).sub(1)
    print()

    # project to combine velocity DoFs
    print()

    # print velocity degrees of freedom
    ndof_velocity = len(up.sub(0).dat.data)
    ndof_pressure = len(up.sub(1).dat.data)
    print(f"number of velocity unknowns  = {ndof_velocity}")
    print(f"number of pressure unknowns  = {ndof_pressure}")
    print(f"---------------------------")
    print(f"total number of unknowns     = {ndof_velocity + ndof_pressure}")
    print()

    # analytic solution
    ud = -plc * ks / mu
    lc = np.sqrt(mue*ks/mu)
    xx = np.linspace(-b, a, nx+1)
    xf = xx[xx>=0]
    xp = xx[xx<0]
    uf = ud - plc*xf*(a-xf)/2/mu - plc*a*lc/2/mue/np.tanh(b/2/lc)
    up = ud - plc*a*lc/2/mue * (np.exp(xp/lc)+np.exp(-(xp+b)/lc)) / (1-np.exp(-b/lc))
    v0, x0 = np.concatenate((up,uf)), np.concatenate((xp,xf))

    # get x coordinates from analytic solution on [-b, a]
    # shift the coordinates so they correspond to [0, a+b]
    # make it "2D" s it corresponds to [0, a+b] x {0}
    true_x_coords_shifted = -(x0+x0[0])[::-1]
    x_coords_2d = np.hstack((true_x_coords_shifted.reshape(-1, 1), np.zeros((true_x_coords_shifted.size, 1))))

    # create a vertex only mesh corresponding to the shifted x coordinates
    vom = VertexOnlyMesh(mesh, x_coords_2d)

    # create a P0DG space over the VOM
    P0DG = FunctionSpace(vom, "DG", 0)

    # input ordering
    P0DG_input_ordering = FunctionSpace(vom.input_ordering, "DG", 0)
    true_io = Function(P0DG_input_ordering)
    true_io.dat.data_wo[:] = -v0[::-1]
    true_sol = assemble(interpolate(true_io, P0DG))

    # print error norm
    error = errornorm(assemble(interpolate(uu, P0DG)), true_sol)/norm(true_sol)
    print(f"Relative L2 error: {error}")

    # plot
    if mg and not smoother=="direct":
        mg_label = "+ mg"
    else:
        mg_label = ""
    fig, ax = plt.subplots()
    ax.plot(x_coord.dat.data, -uu.dat.data, linewidth=0, marker="o", color="cornflowerblue", label=f"{smoother} {mg_label}")
    ax.plot(-(x0+x0[0])[::-1], -v0[::-1], linewidth=2, linestyle="--", color="black", label="Analytic")
    ax.set_xlabel("position (mm)")
    ax.set_ylabel("velocity (mm/s)")
    plt.legend()
    plt.tight_layout()

    a1 = np.sort(true_sol.dat.data)
    a2 = np.sort(-assemble(interpolate(uu, P0DG)).dat.data)

    print(f"Norm?? = {np.linalg.norm(a1-a2)/np.linalg.norm(a1)}")

    # save the figure
    try:
        os.makedirs("parallel_channel_figures", exist_ok=True)
        if mg:
            plt.savefig(f"parallel_channel_figures/pc_{nx}_{smoother}_mg")
        else:
            plt.savefig(f"parallel_channel_figures/pc_{nx}_{smoother}")
    except FileNotFoundError:
        print("Directory does not exist")
    except Exception as e:
        print(f"An error occured while trying to save the figure: {e}")
    finally:
        if show:
            plt.show()

    return None

for k in range(4, 9):
    parallel_channel_solver(nx=2**k, ny=2**k, smoother="braess-sarazin", show=False)



    