"""Deformation of a square droplet. (15 minutes)


_______________________________
|                             |
|                             |
|              0              |
|                             |
|        ___________          |
|        |         |          |
|        |    1    |          |
|        |         |          |
|        |_________|          |
|                             |
|                             |
|                             |
|                             |
|_____________________________|



Initially, two two fluids of the same density are distinguished by a
color index assigned to them and allowed to settle under the effects
of surface tension. It is expected that the surface tension at the
interface between the two fluids deforms the initially square droplet
into a cirular droplet to minimize the interface area/length.

The references for this problem are

 - J. Morris "Simulating surface tension with smoothed particle
   hydrodynamics", 2000, IJNMF, 33, pp 333--353 [JM00]

 - S. Adami, X.Y. Hu, N.A. Adams "A new surface tension formulation
   for multi-phase SPH using a reproducing divergence approximation",
   2010, JCP, 229, pp 5011--5021 [AHA10]

 - M. S. Shadloo, M. Yildiz "Numerical modelling of Kelvin-Helmholtz
   instability using smoothed particle hydrodynamics", IJNME, 2011,
   87, pp 988--1006 [SY11]

The surface-tension model used currently is the CSF model based on
interface curvature and normals computed from the color function.

"""
import numpy

import numpy as np
import os

from surface_tension import MomentumEquationViscosityAdami, StateEquation, AdamiReproducingDivergence, CSFSurfaceTensionForceAdami

from surface_tension import SummationDensity, MomentumEquationPressureGradientAdami, MomentumEquationViscosityAdami, SolidWallPressureBCnoDensity, ColorGradientAdami, ConstructStressMatrix, SurfaceForceAdami

from pysph.sph.wc.transport_velocity import SummationDensity, \
    MomentumEquationPressureGradient,\
    SolidWallPressureBC, SolidWallNoSlipBC, StateEquation,\
    MomentumEquationArtificialStress, MomentumEquationViscosity
from pysph.sph.surface_tension import InterfaceCurvatureFromNumberDensity, \
    ShadlooYildizSurfaceTensionForce, CSFSurfaceTensionForce, \
    SmoothedColor, AdamiColorGradient, MorrisColorGradient, \
    SY11DiracDelta, AdamiReproducingDivergence, SY11ColorGradient

from pysph.sph.wc.basic import TaitEOS
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.tools.geometry import get_2d_block
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.sph.equation import Group, Equation

from pysph.sph.surface_tension import SmoothedColor, CSFSurfaceTensionForce

from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator, EPECIntegrator

from pysph.base.nnps import DomainManager
from pysph.solver.utils import iter_output
from surface_tension import SummationDensitySourceMass, MomentumEquationViscosityMorris, MomentumEquationPressureGradientMorris, InterfaceCurvatureFromDensity, MorrisColorGradient

# Particle generator
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, WendlandQuintic, Gaussian, \
    QuinticSpline

# SPH Equations and Group
from pysph.sph.equation import Group

from pysph.sph.wc.viscosity import ClearyArtificialViscosity

from pysph.sph.wc.transport_velocity import SummationDensity, \
    MomentumEquationPressureGradient,\
    SolidWallPressureBC, SolidWallNoSlipBC, StateEquation,\
    MomentumEquationArtificialStress, MomentumEquationViscosity

from pysph.sph.surface_tension import InterfaceCurvatureFromNumberDensity, \
    ShadlooYildizSurfaceTensionForce, CSFSurfaceTensionForce, \
    SmoothedColor, AdamiColorGradient, MorrisColorGradient, \
    SY11DiracDelta, AdamiReproducingDivergence, SY11ColorGradient

from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

# PySPH solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# Integrators and Steppers
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

# Domain manager for periodic domains
from pysph.base.nnps import DomainManager

# problem parameters
dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
sigma = 1.0

# set factor1 to [0.5 ~ 1.0] to simulate a thick or thin
# interface. Larger values result in a thick interface.
factor1 = 0.8
factor2 = 1./factor1

# discretization parameters
dx = dy = 0.0125
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.3
h0 = hdx * dx
rho0 = 1
c0 = 20.0
p0 = c0*c0*rho0
nu = 0.2

# correction factor for Morris's Method I. Set with_morris_correction
# to True when using this correction.
epsilon = 0.01/h0

# time steps
tf = 1.0
dt_cfl = 0.25 * h0/(1.1*c0)
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

dt = 0.9 * min(dt_cfl, dt_viscous, dt_force)


class SquareDroplet(Application):
    def add_user_options(self, group):
        choices = ['morris', 'tvf', 'adami_stress', 'adami', 'shadloo']
        group.add_argument(
            "--scheme", action="store", dest='scheme', default='morris',
            choices=choices,
            help='Specify scheme to use among %s' % choices
        )

    def create_particles(self):
        x, y = numpy.mgrid[dxb2:domain_width:dx, dyb2:domain_height:dy]
        x = x.ravel()
        y = y.ravel()

        m = numpy.ones_like(x) * volume * rho0
        rho = numpy.ones_like(x) * rho0
        h = numpy.ones_like(x) * h0
        cs = numpy.ones_like(x) * c0

        # additional properties required for the fluid.
        additional_props = [
            # volume inverse or number density
            'V', 'alpha',

            # color and gradients
            'color', 'scolor', 'cx', 'cy', 'cz', 'cx2', 'cy2', 'cz2',

            # discretized interface normals and dirac delta
            'nx', 'ny', 'nz', 'ddelta',

            # interface curvature
            'kappa',

            # transport velocities
            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',

            # imposed accelerations on the solid wall
            'ax', 'ay', 'az', 'wij',

            # velocity of magnitude squared
            'vmag2',

            # variable to indicate reliable normals and normalizing
            # constant
            'N', 'wij_sum', 'pi00', 'pi01', 'pi10', 'pi11', 'nu'

            ]

        # get the fluid particle array
        fluid = get_particle_array(
            name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs,
            additional_props=additional_props)

        # set the color of the inner square
        for i in range(x.size):
            if ((fluid.x[i] > 0.35) and (fluid.x[i] < 0.65)):
                if ((fluid.y[i] > 0.35) and (fluid.y[i] < 0.65)):
                    fluid.color[i] = 1.0

        # particle volume
        fluid.V[:] = 1./volume
        fluid.nu[:] = nu
        fluid.alpha[:] = sigma
        # set additional output arrays for the fluid
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                 'ddelta', 'kappa', 'N', 'scolor', 'p'])

        print("2D Square droplet deformation with %d fluid particles" % (
                fluid.get_number_of_particles()))

        return [fluid, ]

    def create_domain(self):
        return DomainManager(
            xmin=0, xmax=domain_width, ymin=0, ymax=domain_height,
            periodic_in_x=True, periodic_in_y=True)

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        sy11_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SY11ColorGradient(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor1)
            ], real=False, update_nnps=True),
            Group(equations=[
                SY11DiracDelta(dest='fluid', sources=['fluid'])
            ], real=False
            ),
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid'],
                    with_morris_correction=True),
            ], real=False),
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor2)
            ], real=False, update_nnps=True,
            ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),
                    ShadlooYildizSurfaceTensionForce(dest='fluid',
                                                     sources=None,
                                                     sigma=sigma),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid']),
                ], )
        ]

        adami_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
            ], real=False),
            Group(equations=[
                AdamiColorGradient(dest='fluid', sources=['fluid']),
            ], real=False
            ),
            Group(equations=[
                AdamiReproducingDivergence(dest='fluid', sources=['fluid'],
                                           dim=2),
            ], real=False),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),
                    MomentumEquationViscosityAdami(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForceAdami(dest='fluid', sources=None,)
                ], )
        ]

        adami_stress_equations = [
            Group(equations=[
                SummationDensity(
                    dest='fluid', sources=[
                        'fluid']),
            ], real=False),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None,
                        rho0=rho0, c0=c0, gamma=7, p0=p0),
            ], real=False),
            Group(equations=[
                ColorGradientAdami(dest='fluid', sources=['fluid']),
            ], real=False),
            Group(equations=[ConstructStressMatrix(
                dest='fluid', sources=None, sigma=sigma, d=2)], real=False),
            Group(
                equations=[
                    MomentumEquationPressureGradientAdami(
                        dest='fluid', sources=['fluid']),
                    MomentumEquationViscosityAdami(
                        dest='fluid', sources=['fluid']),
                    SurfaceForceAdami(
                        dest='fluid', sources=['fluid']),
                ]),
        ]

        tvf_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ], real=False),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SmoothedColor(dest='fluid', sources=['fluid']),
            ], real=False),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid'],
                                    epsilon=epsilon),
            ], real=False
            ),
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid'],
                    with_morris_correction=True),
            ], real=False),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),
                    CSFSurfaceTensionForce(dest='fluid', sources=None,
                                           sigma=sigma),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid']),
                ], )
        ]

        morris_equations = [
            Group(equations=[
                SummationDensitySourceMass(
                    dest='fluid', sources=[
                        'fluid']),
            ], real=False, update_nnps=False),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None,
                        rho0=rho0, c0=c0, gamma=1.0),
                SmoothedColor(
                    dest='fluid', sources=['fluid', ]),
            ], real=False, update_nnps=False),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid', ],
                                    epsilon=epsilon),
            ], real=False, update_nnps=False),
            Group(equations=[
                InterfaceCurvatureFromDensity(dest='fluid', sources=['fluid'],
                                              with_morris_correction=True),
            ], real=False, update_nnps=False),
            Group(
                equations=[
                    MomentumEquationPressureGradientMorris(
                        dest='fluid', sources=['fluid']),
                    MomentumEquationViscosityMorris(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForce(
                        dest='fluid', sources=None, sigma=sigma),
                ], update_nnps=False)
        ]

        if self.options.scheme == 'tvf':
            return tvf_equations
        elif self.options.scheme == 'adami_stress':
            return adami_stress_equations
        elif self.options.scheme == 'adami':
            return adami_equations
        elif self.options.scheme == 'shadloo':
            return sy11_equations
        else:
            return morris_equations


if __name__ == '__main__':
    app = SquareDroplet()
    app.run()
