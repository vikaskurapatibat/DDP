import numpy
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, WendlandQuintic, Gaussian, QuinticSpline
from pysph.sph.equation import Group
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator
from pysph.base.nnps import DomainManager
from pysph.sph.wc.transport_velocity import SummationDensity, MomentumEquationPressureGradient
from pysph.sph.equation import Equation
from pysph.sph.surface_tension import AdamiColorGradient

dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
sigma = 1.0
eta = 0.2

# discretization parameters
dx = dy = 0.0125
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.3
h0 = hdx * dx
rho0 = 1.0
c0 = 20.0
p0 = c0*c0*rho0
nu = eta/rho0

# correction factor for Morris's Method I. Set with_morris_correction
# to True when using this correction.
epsilon = 0.01/h0

# time steps
tf = 1.0
dt_cfl = 0.25 * h0/(1.1*c0)
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

dt = 0.9 * min(dt_cfl, dt_viscous, dt_force)


class MomentumEquationViscosity(Equation):
    def loop(self, d_idx, s_idx, d_nu, s_nu, d_m, d_V, s_V, VIJ, XIJ, DWIJ, R2IJ, EPS, d_au, d_av, d_aw):
        Vi = 1/d_V[d_idx]
        Vj = 1/s_V[s_idx]
        factor = 2*d_nu[d_idx]*s_nu[s_idx]/(d_nu[d_idx]+s_nu[s_idx])*(Vi*Vi + Vj*Vj)
        FIJ = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]
        tmp = FIJ*factor/(R2IJ+EPS)
        d_au[d_idx] += tmp*VIJ[0]/d_m[d_idx]
        d_av[d_idx] += tmp*VIJ[1]/d_m[d_idx]
        d_aw[d_idx] += tmp*VIJ[2]/d_m[d_idx]


class StateEquation(Equation):

    def __init__(self, dest, sources, p0, rho0, gamma=7):
        self.gamma = gamma
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        factor = (d_rho[d_idx]/self.rho0)**(self.gamma)
        d_p[d_idx] = self.p0*(factor - 1) + self.p0


class AdamiReproducingDivergence(Equation):

    def __init__(self, dest, sources, dim):
        self.dim = dim
        super(AdamiReproducingDivergence, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa, d_wij_sum):
        d_kappa[d_idx] = 0.0
        d_wij_sum[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_wij_sum,
             d_nx, d_ny, d_nz, s_nx, s_ny, s_nz, d_V, s_V,
             DWIJ, XIJ, RIJ, EPS):
        # particle volumes
        Vi = 1./d_V[d_idx]
        Vj = 1./s_V[s_idx]

        # dot product in the numerator of Eq. (20)
        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]

        # dot product in the denominator of Eq. (20)
        xijdotdwij = XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]

        # accumulate the contributions
        d_kappa[d_idx] += nijdotdwij * Vj
        d_wij_sum[d_idx] += xijdotdwij * Vj

    def post_loop(self, d_idx, d_kappa, d_wij_sum):
        # normalize the curvature estimate
        d_kappa[d_idx] /= d_wij_sum[d_idx]
        d_kappa[d_idx] *= -self.dim


class CSFSurfaceTensionForce(Equation):
    
    def loop(self, d_idx, d_au, d_av, d_aw, d_kappa, d_cx, d_cy, d_cz, d_m, d_alpha):
        d_au[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cx[d_idx]/d_m[d_idx]
        d_av[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cy[d_idx]/d_m[d_idx]
        d_aw[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cz[d_idx]/d_m[d_idx]


class Adami(Application):
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
            'V',

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
            'N', 'wij_sum', 'nu', 'alpha'

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
        adami_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid'])
            ]),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
            ]),
            Group(equations=[
                AdamiColorGradient(dest='fluid', sources=['fluid']),
            ],
            ),
            Group(equations=[
                AdamiReproducingDivergence(dest='fluid', sources=['fluid'],
                                           dim=2),
            ], ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid'], pb=p0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForce(dest='fluid', sources=None,)
                ], )
        ]

        return adami_equations

    def pre_step(self, solver):
        solver.dump_output()


if __name__ == '__main__':
    app = Adami()
    app.run()
