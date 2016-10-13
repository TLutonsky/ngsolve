from netgen.geom2d import unit_square
from ngsolve import *

mesh = Mesh (unit_square.GenerateMesh(maxh=0.1))

fes = L2(mesh, order=4, flags = { "dgjumps" : True } )

u = fes.TrialFunction()
v = fes.TestFunction()

b = CoefficientFunction( (y-0.5,0.5-x) )
bn = b*specialcf.normal(2)

ubnd = CoefficientFunction(0)

a = BilinearForm(fes)
a += SymbolicBFI ( (-u * b*grad(v) ).Compile(True) )

# the skeleton-formulation, sum over edges:
a += SymbolicBFI ( (bn*IfPos(bn, u, u.Other()) * (v-v.Other())).Compile(True), VOL, skeleton=True)
a += SymbolicBFI ( (bn*IfPos(bn, u, ubnd) * v), BND, skeleton=True)

# or the element-boundary formulation
# note the bnd-value in the .Other operator
# a += SymbolicBFI ( (bn*IfPos(bn, u, u.Other(bnd=ubnd)) * v), element_boundary=True)

u = GridFunction(fes)
u.Set(exp (-40 * ( (x-0.7)*(x-0.7) + (y-0.7)*(y-0.7) )))

w = u.vec.CreateVector()

Draw (u, autoscale=False, sd=2)

t = 0
tau = 1e-3
tend = 10

with TaskManager():
    while t < tend:
        a.Apply (u.vec, w)
        fes.SolveM (rho=CoefficientFunction(1), vec=w)
        u.vec.data -= tau * w
        t += tau
        Redraw()
    
