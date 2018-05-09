import numpy

# Particle generator
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, WendlandQuintic, Gaussian, \
    QuinticSpline
from pysph.sph.equation import Equation

# SPH Equations and Group
from pysph.sph.equation import Group

from pysph.sph.wc.viscosity import ClearyArtificialViscosity

from pysph.sph.wc.transport_velocity import SummationDensity, \
    SolidWallPressureBC, SolidWallNoSlipBC, StateEquation,\
    MomentumEquationArtificialStress, MomentumEquationViscosity

from pysph.sph.surface_tension import ShadlooYildizSurfaceTensionForce, CSFSurfaceTensionForce, \
    SmoothedColor, AdamiColorGradient, MorrisColorGradient, \
    AdamiReproducingDivergence, SY11ColorGradient

from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

# PySPH solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# Integrators and Steppers
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator_step import VerletSymplecticWCSPHStep

from pysph.sph.integrator import PECIntegrator
from pysph.sph.basic_equations import XSPHCorrection, IsothermalEOS, BodyForce

# Domain manager for periodic domains
from pysph.base.nnps import DomainManager

# problem parameters
dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
alpha = 0.1
wavelength = 1.0
wavenumber = 2*numpy.pi/wavelength
Ri = 0.1
rho0 = rho1 = 1000.0
rho2 = rho1
U = 0.5
sigma = Ri * (rho1*rho2) * (2*U)**2/(wavenumber*(rho1 + rho2))
psi0 = 0.03*domain_height
gy = -9.81

# discretization parameters
nghost_layers = 5
dx = dy = 0.0125
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.3
h0 = hdx * dx
rho0 = 1000.0
c0 = 10.0
p0 = c0*c0*rho0
nu = 0.125 * alpha * h0 * c0

# time steps and final time
tf = 3.0
dt = 1e-4

factor1 = 0.8
factor2 = 1/factor1


class SY11DiracDelta(Equation):
    r"""Discretized dirac-delta for the SY11 formulation Eq. (14) in [SY11]

    This is essentially the same as computing the color gradient, the
    only difference being that this might be called with a reduced
    smoothing length.

    Note that the normals should be computed using the
    SY11ColorGradient equation. This function will effectively
    overwrite the color gradient.

    """

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon*epsilon
        super(SY11DiracDelta, self).__init__(dest, sources)

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_ddelta, d_nx, d_ny, d_nz, d_N):
        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # discretized dirac delta
        d_ddelta[d_idx] = 0.0
        d_N[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_color, s_color, d_cx, d_cy, d_cz,
             d_V, s_V, DWIJ):

        # average particle volume
        psiab1 = 2.0/(d_V[d_idx] + s_V[s_idx])

        # difference in color divided by psiab. Eq. (13) in [SY11]
        Cba = (s_color[s_idx] - d_color[d_idx]) * psiab1

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):
        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        if mod_gradc2 > self.epsilon2:
            mod_gradc = sqrt(mod_gradc2)

            # d_N[d_idx] = 1.0

            d_nx[d_idx] = d_cx[d_idx]/mod_gradc
            d_ny[d_idx] = d_cy[d_idx]/mod_gradc
            d_nz[d_idx] = d_cz[d_idx]/mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = mod_gradc


class InterfaceCurvatureFromNumberDensity(Equation):
    def initialize(self, d_idx, d_kappa):
        d_kappa[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_nx, s_nx, d_ny, s_ny, d_nz, s_nz, d_V, s_V, 
             DWIJ, d_kappa, d_N, s_N):
        psiab = 2.0/(d_V[d_idx]+s_V[s_idx])
        nijdotwij = (d_nx[d_idx]-s_nx[s_idx])*DWIJ[0] + (d_ny[d_idx]-s_ny[s_idx])*DWIJ[1] + (d_nz[d_idx]-s_nz[s_idx])*DWIJ[2]
        tmp = 1.0
        tmp = min(d_N[d_idx], s_N[s_idx])
        d_kappa[d_idx] += tmp*psiab*nijdotwij


class MomentumEquationPressureGradient(Equation):
    def initiliaze(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, DWIJ, d_au, d_av, d_aw, d_V, s_V, d_p, s_p):
        pi = d_p[d_idx]/(d_V[d_idx]*d_V[d_idx])
        pj = s_p[s_idx]/(s_V[s_idx]*s_V[s_idx])
        factor = pi+pj
        d_au[d_idx] += (pi+pj)*DWIJ[0]
        d_av[d_idx] += (pi+pj)*DWIJ[1]
        d_aw[d_idx] += (pi+pj)*DWIJ[2]


class ShadlooArtificialViscosity(Equation):
    def __init__(self, dest, sources, alpha=1.0):
        self.alpha = alpha
        super(ShadlooArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_h, s_h, d_cs, s_cs, d_rho, s_rho, VIJ, XIJ,
             R2IJ, EPS, d_V, s_V, d_au, d_av, d_aw, DWIJ):
        mua = 0.125*self.alpha*d_h[d_idx]*d_cs[d_idx]*d_rho[d_idx]
        mub = 0.125*self.alpha*s_h[s_idx]*s_cs[s_idx]*s_rho[s_idx]
        muab = 2.0*mua*mub/(mua+mub)
        vijdotxij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        den = d_V[d_idx]*s_V[s_idx]*(R2IJ + EPS)
        piij = 8.0*muab*vijdotxij/den
        d_au[d_idx] += piij*DWIJ[0]
        d_av[d_idx] += piij*DWIJ[1]
        d_aw[d_idx] += piij*DWIJ[2]


class SquareDroplet(Application):
    def create_particles(self):
        ghost_extent = (nghost_layers + 0.5)*dx

        x, y = numpy.mgrid[dxb2:domain_width:dx,
                           -ghost_extent:domain_height+ghost_extent:dy]
        x = x.ravel()
        y = y.ravel()

        m = numpy.ones_like(x) * volume * rho0
        rho = numpy.ones_like(x) * rho0
        p = numpy.ones_like(x) * p0
        h = numpy.ones_like(x) * h0
        cs = numpy.ones_like(x) * c0

        # additional properties required for the fluid.
        additional_props = [
            # volume inverse or number density
            'V',

            # color and gradients
            'color', 'scolor', 'cx', 'cy', 'cz', 'cx2', 'cy2', 'cz2',

            # discretized interface normals and dirac delta
            'nx', 'ny', 'nz', 'ddelta',

            # interface curvature
            'kappa',

            # filtered velocities
            'uf', 'vf', 'wf',

            # transport velocities
            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',

            # imposed accelerations on the solid wall
            'ax', 'ay', 'az', 'wij',

            # velocity of magnitude squared
            'vmag2',

            # variable to indicate reliable normals and normalizing
            # constant
            'N', 'wij_sum',

        ]

        # get the fluid particle array
        fluid = get_particle_array(
            name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs, p=p,
            additional_props=additional_props)

        # set the fluid velocity with respect to the sinusoidal
        # perturbation
        fluid.u[:] = -U
        mode = 1
        for i in range(len(fluid.x)):
            ang = 2*numpy.pi*fluid.x[i]/(mode*domain_width)
            if fluid.y[i] > domain_height/2+psi0*domain_height*numpy.sin(ang):
                fluid.u[i] = U
                fluid.color[i] = 1.0

        # extract the top and bottom boundary particles
        indices = numpy.where(fluid.y > domain_height)[0]
        wall = fluid.extract_particles(indices)
        fluid.remove_particles(indices)

        indices = numpy.where(fluid.y < 0)[0]
        bottom = fluid.extract_particles(indices)
        fluid.remove_particles(indices)

        # concatenate the two boundaries
        wall.append_parray(bottom)
        wall.set_name('wall')

        # set the number density initially for all particles
        fluid.V[:] = 1./volume
        wall.V[:] = 1./volume
        for i in range(len(wall.x)):
            if wall.y[i] > 0.5:
                wall.color[i] = 1.0
            else:
                wall.color[i] = 0.0
        # set additional output arrays for the fluid
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                 'ddelta', 'p', 'rho', 'au', 'av'])
        wall.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny',
                                'ddelta', 'p', 'rho', 'au', 'av'])

        print("2D KHI with %d fluid particles and %d wall particles" % (
            fluid.get_number_of_particles(),
            wall.get_number_of_particles()))

        return [fluid, wall]

    def create_domain(self):
        return DomainManager(xmin=0, xmax=domain_width, ymin=0,
                             ymax=domain_height,
                             periodic_in_x=True, periodic_in_y=False)

    def create_solver(self):
        kernel = WendlandQuintic(dim=2)
        integrator = PECIntegrator(fluid=VerletSymplecticWCSPHStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        sy11_equations = [
            # We first compute the mass and number density of the fluid
            # phase. This is used in all force computations henceforth. The
            # number density (1/volume) is explicitly set for the solid phase
            # and this isn't modified for the simulation.
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'wall']),
                SummationDensity(dest='wall', sources=['fluid', 'wall'])
            ], real=False),

            # Given the updated number density for the fluid, we can update
            # the fluid pressure. Additionally, we can compute the Shepard
            # Filtered velocity required for the no-penetration boundary
            # condition. Also compute the gradient of the color function to
            # compute the normal at the interface.
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
            ], real=False),

            Group(
                equations=[
                    SolidWallPressureBC(
                        dest='wall', sources=['fluid'], p0=p0, rho0=rho0,
                        gy=gy, b=1.0),
                ], real=False),


            #################################################################
            # Begin Surface tension formulation
            #################################################################
            # Scale the smoothing lengths to determine the interface
            # quantities.
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor1)
            ], update_nnps=True),

            # Compute the discretized dirac delta with respect to the new
            # smoothing length.
            Group(equations=[
                SY11DiracDelta(dest='fluid', sources=['fluid'])
            ], real=False),

            # Compute the interface curvature using the modified smoothing
            # length and interface normals computed in the previous Group.
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid']),
            ], real=False),

            # Now rescale the smoothing length to the original value for the
            # rest of the computations.
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor2)
            ], update_nnps=True),
            #################################################################
            # End Surface tension formulation
            #################################################################

            # The main acceleration block
            Group(
                equations=[

                    # Gradient of pressure for the fluid phase using the
                    # number density formulation. No penetration boundary
                    # condition using Adami et al's generalized wall boundary
                    # condition. The extrapolated pressure and density on the
                    # wall particles is used in the gradient of pressure to
                    # simulate a repulsive force.
                    BodyForce(dest='fluid', sources=None, fy=gy),

                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'wall']),

                    # Artificial viscosity for the fluid phase.
                    ShadlooArtificialViscosity(dest='fluid', sources=['fluid']),

                    # Surface tension force for the SY11 formulation
                    ShadlooYildizSurfaceTensionForce(dest='fluid',
                                                     sources=None,
                                                     sigma=sigma),
                ], )
        ]

        return sy11_equations


if __name__ == '__main__':
    app = SquareDroplet()
    app.run()
