from firedrake import *
from firedrake.petsc import PETSc

'''
Firedrake demo for the (homogeneous) Stokes equations:

-div(grad(u)) + grad(p) = 0
                div(u)  = 0

On:

[0, 1] x [0, 1]

Subject to:

u(0, y) = (0, 0)
u(1, y) = (0, 0)
u(x, 0) = (0, 0)
u(x, 1) = (1, 0)

'''

# create mesh
N = 10
mesh = UnitSquareMesh(N, N)

# create function spaces (Taylor Hood)
# V is a function space of vector valued functions using Lagrange finite elements of order 2
# Q is a function space of scalar valued functions using Lagrange finite elements of order 1
# X is the combination of these two spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
X = V * Q

# create trial and test functions
# u \in V, p \in Q
# v \in V, q \in Q
u, p = TrialFunctions(X)
v, q = TestFunctions(X)

# boundary conditions as specified in docstring at start of file
# they only apply to the velocity space i.e. X.sub(0)
bcs = [DirichletBC(X.sub(0), Constant((1, 0)), (4,)), 
       DirichletBC(X.sub(0), Constant((0, 0)), (1, 2, 3))]

# define bilinear form
a = (inner(grad(u), grad(v)) + inner(p, div(v)) - inner(q, div(u))) * dx
# define cts. linear functional
# since we are in the homogeneous case, f = Constant(0, 0)
L = inner(Constant((0,0)), v) * dx

# in the non-hom case would have f = (F, 0)

# no boundary conditions for pressure
# but pressure is only defined up to a constant
# need a way to get a unique pressure solution
nullspace = MixedVectorSpaceBasis(X, [X.sub(0), VectorSpaceBasis(constant=True)])

# solve
up = Function(X)

try:
    solve(a == L, up, bcs=bcs, nullspace=nullspace,
          solver_parameters={"ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
except PETSc.Error as e:
    if e.ierr == 92:
        warning("MUMPS not installed, skipping direct solve")
    else:
        raise e

# extract velocity and pressure solutions
u, p = up.subfunctions

# write to a file

# plotting

try:
    import matplotlib.pyplot as plt
except:
    warning("Couldn't import matplotlib")

try:
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    fig.colorbar(colors)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

try:
    plt.show()
except Exception as e:
    warning("Cannot show figure. Error msg: '%s'" % e)
