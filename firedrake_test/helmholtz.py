from firedrake import *

'''
Firedrake tutorial for the Helmholtz equation (in meteorology)
'''

# create mesh
mesh = UnitSquareMesh(10, 10)

# define function space
V = FunctionSpace(mesh, "CG", 1)

# test and trial functions
u = TrialFunction(V)
v = TestFunction(V)

# define RHS function
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

# define bilinear form and cts. linear functional
a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
L = inner(f, v) * dx

# solve equation
u = Function(V)
solve(a == L, u, solver_parameters={"ksp_type": "cg", "pc_type": "none"})

# output to a file
File("helmholtz.pvd").write(u)

# import matplotlib
try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

# pseudocolour plot
try:
    fig, axes = plt.subplots()
    colors = tripcolor(u, axes=axes)
    fig.colorbar(colors)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

# contour plot
try:
    fig, axes = plt.subplots()
    contours = tricontour(u, axes=axes)
    fig.colorbar(contours)
except Exception as e:
    warning("Cannot plot figure. Error msg: '%s'" % e)

# show figure
try:
    plt.show()
except Exception as e:
    warning("Cannot show figure. Error msg: '%s'" % e)

# compare to analytic solution in L2 norm
f.interpolate(cos(x*pi*2)*cos(y*pi*2))
print("L2 error: {}".format(sqrt(assemble(dot(u - f, u - f) * dx))))
