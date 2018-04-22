
import numpy as np
import os

from math import sqrt
from pysph.solver.utils import load
from pysph.sph.wc.basic import TaitEOS

from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.tools.geometry import get_2d_block, get_2d_tank, get_2d_wall, remove_overlap_particles
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.sph.equation import Group, Equation

from pysph.sph.surface_tension import SmoothedColor, CSFSurfaceTensionForce

from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.wc.transport_velocity import SolidWallNoSlipBC
from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import EPECIntegrator

# from surface_tension import SolidWallPressureBCnoDensity, SummationDensitySourceMass, MomentumEquationViscosityMorris, \
# MomentumEquationPressureGradientMorris, MorrisColorGradient, InterfaceCurvatureFromDensity


class InterfaceCurvatureFromDensity(Equation):

    def __init__(self, dest, sources, with_morris_correction=True):
        self.with_morris_correction = with_morris_correction

        super(InterfaceCurvatureFromDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kappa, d_wij_sum):
        d_kappa[d_idx] = 0.0
        d_wij_sum[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_kappa, d_nx, d_ny, d_nz, s_nx, s_ny, s_nz,
             d_V, s_V, d_N, s_N, d_wij_sum, s_rho, s_m, WIJ, DWIJ):

        nijdotdwij = (d_nx[d_idx] - s_nx[s_idx]) * DWIJ[0] + \
            (d_ny[d_idx] - s_ny[s_idx]) * DWIJ[1] + \
            (d_nz[d_idx] - s_nz[s_idx]) * DWIJ[2]

        tmp = 1.0
        if self.with_morris_correction:
            tmp = min(d_N[d_idx], s_N[s_idx])

        d_wij_sum[d_idx] += tmp * s_m[s_idx]/s_rho[s_idx] * WIJ

        d_kappa[d_idx] += tmp*nijdotdwij*s_m[s_idx]/s_rho[s_idx]

    def post_loop(self, d_idx, d_wij_sum, d_nx, d_kappa):

        if self.with_morris_correction:
            if d_wij_sum[d_idx] > 1e-12:
                d_kappa[d_idx] /= d_wij_sum[d_idx]


class MorrisColorGradient(Equation):

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon*epsilon
        super(MorrisColorGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_cx, d_cy, d_cz, d_nx, d_ny, d_nz,
                   d_ddelta, d_N):

        # color gradient
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

        # interface normals
        d_nx[d_idx] = 0.0
        d_ny[d_idx] = 0.0
        d_nz[d_idx] = 0.0

        # reliability indicator for normals and dirac delta
        d_N[d_idx] = 0.0
        d_ddelta[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_scolor, s_scolor, d_cx, d_cy, d_cz,
             s_m, s_rho, DWIJ):

        # Eq. (17) in [JM00]
        Cba = (s_scolor[s_idx] - d_scolor[d_idx]) * s_m[s_idx]/s_rho[s_idx]

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):

        # this is to avoid the very small values of
        # normal direction which give spurious results
        # of derivatives which build on and give unstable
        # results at the interfaces
        # if d_cx[d_idx]*d_cx[d_idx] < self.epsilon2:
        #     d_cx[d_idx] = 0.0
        # if d_cy[d_idx]*d_cy[d_idx] < self.epsilon2:
        #     d_cy[d_idx] = 0.0
        # if d_cz[d_idx]*d_cz[d_idx] < self.epsilon2:
        #     d_cz[d_idx] = 0.0

        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + \
            d_cy[d_idx]*d_cy[d_idx] + \
            d_cz[d_idx]*d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        # (particles for which the color gradient is zero) Eq. (19,
        # 20) in [JM00]
        if mod_gradc2 > self.epsilon2:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            mod_gradc = 1./sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = 1./mod_gradc


class MomentumEquationPressureGradientMorris(Equation):

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_p, s_p, DWIJ, d_rho, s_rho):
        factor = -s_m[s_idx]*(d_p[d_idx] + s_p[s_idx]) / \
            (d_rho[d_idx]*s_rho[s_idx])
        d_au[d_idx] += factor*DWIJ[0]
        d_av[d_idx] += factor*DWIJ[1]
        d_aw[d_idx] += factor*DWIJ[2]


class SolidWallPressureBCnoDensity(Equation):

    def initialize(self, d_idx, d_p, d_wij):
        d_p[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, d_wij, s_rho,
             d_au, d_av, d_aw, WIJ, XIJ):

        d_p[d_idx] += s_p[s_idx]*WIJ

        d_wij[d_idx] += WIJ

    def post_loop(self, d_idx, d_wij, d_p, d_rho):
        if d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]


class SummationDensitySourceMass(Equation):

    def initialize(self, d_idx, d_V, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ, s_m, s_idx):
        d_rho[d_idx] += s_m[s_idx]*WIJ

    def post_loop(self, d_idx, d_V, d_rho, d_m):
        d_V[d_idx] = d_rho[d_idx]/d_m[d_idx]


class MomentumEquationViscosityMorris(Equation):

    def __init__(self, dest, sources, eta=0.01):
        self.eta = eta*eta
        super(MomentumEquationViscosityMorris, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_nu, s_nu, d_rho,
             s_rho, DWIJ, R2IJ, VIJ, HIJ, XIJ):

        dw = (DWIJ[0]*XIJ[0]+DWIJ[1]*XIJ[1]+DWIJ[2]
              * XIJ[2])/(R2IJ+self.eta*HIJ*HIJ)
        mult = s_m[s_idx]*(d_nu[d_idx]+s_nu[s_idx])/(d_rho[d_idx]*s_rho[s_idx])
        d_au[d_idx] += dw*mult*VIJ[0]
        d_av[d_idx] += dw*mult*VIJ[1]
        d_aw[d_idx] += dw*mult*VIJ[2]


dim = 2
Lx = 1.0
Ly = 1.0

nu0 = 0.05
sigma = 1.0
factor1 = 0.5
factor2 = 1 / factor1
rho1 = 1.0

c0 = 20.0
gamma = 1.4
R = 287.1

p1 = c0**2 * rho1

nx = 120
dx = Lx / nx
volume = dx * dx
hdx = 1.5

h0 = hdx * dx

tf = 0.5

epsilon = 0.01 / h0

r0 = 0.05
v0 = 10.0


dt1 = 0.25*np.sqrt(rho1*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0)

dt3 = 0.125*rho1*h0*h0/nu0

dt = 0.9*min(dt1, dt2, dt3)


def r(x, y):
    return x*x + y*y


class MultiPhase(Application):
    global nu0

    def create_particles(self):
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx, height=Ly, center=np.array([0., 0.]))
        rho_fluid = np.ones_like(fluid_x) * rho1
        m_fluid = rho_fluid * volume
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        wall_x, wall_y = get_2d_block(dx=dx, length=Lx+6*dx, height=Ly+6*dx, center = np.array([0., 0.]))
        rho_wall = np.ones_like(wall_x) * rho1
        m_wall = rho_wall * volume
        h_wall = np.ones_like(wall_x) * h0
        cs_wall = np.ones_like(wall_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu', 'wg', 'ug', 'vg',
                            'pi00', 'pi01', 'pi10', 'pi11']
        consts = {'max_ddelta': np.zeros(1, dtype=float)}
        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props, constants=consts)
        for i in range(len(fluid.x)):
            if (fluid.x[i]*fluid.x[i] + fluid.y[i]*fluid.y[i]) < 0.1875*0.1875:
                fluid.color[i] = 1.0
            else:
                fluid.color[i] = 0.0
        wall = get_particle_array(
            name='wall', x=wall_x, y=wall_y, h=h_wall, m=m_wall,
            rho=rho_wall, cs=cs_wall, additional_props=additional_props)
        wall.color[:] = 0.0
        remove_overlap_particles(wall, fluid, dx_solid=dx, dim=2)
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                 'kappa', 'N', 'scolor', 'p'])
        wall.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                 'kappa', 'N', 'scolor', 'p'])
        for i in range(len(fluid.x)):
            R = sqrt(r(fluid.x[i], fluid.y[i]) + 0.0001*fluid.h[i]*fluid.h[i])
            fluid.u[i] = v0*fluid.x[i]*(1.0 - (fluid.y[i]*fluid.y[i])/(r0*R))*np.exp(-R/r0)/r0
            fluid.v[i] = -v0*fluid.y[i]*(1.0 - (fluid.x[i]*fluid.x[i])/(r0*R))*np.exp(-R/r0)/r0
        fluid.nu[:] = nu0

        return [fluid, wall]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = EPECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=1,
            output_at_times=[0., 0.08, 0.16, 0.26])
        return solver

    def create_equations(self):
        morris_equations = [
            Group(equations=[
                SummationDensitySourceMass(
                    dest='fluid', sources=[
                        'fluid', 'wall']),
                SummationDensitySourceMass(
                    dest='wall', sources=[
                        'fluid', 'wall']),
            ]),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=rho1, c0=c0, gamma=7, p0=0.0),
                SolidWallPressureBCnoDensity(dest='wall', sources=['fluid']),
                SmoothedColor(
                    dest='fluid', sources=[
                        'fluid', 'wall']),
                SmoothedColor(
                    dest='wall', sources=[
                        'fluid', 'wall']),
            ]),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid', 'wall'],
                                    epsilon=epsilon),
            ]),
            Group(equations=[
                InterfaceCurvatureFromDensity(dest='fluid', sources=['fluid', 'wall'],
                                                    with_morris_correction=True),
            ]),
            Group(
                equations=[
                    MomentumEquationPressureGradientMorris(
                        dest='fluid', sources=['fluid', 'wall']),
                    MomentumEquationViscosityMorris(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForce(
                        dest='fluid', sources=None, sigma=sigma),
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu0)
                ]),
                ]

        return morris_equations


if __name__ == '__main__':
    app = MultiPhase()
    app.run()


# Generalised TVF 2017 jcp Hu and Adams
