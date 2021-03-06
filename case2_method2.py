import numpy as np
import os

from pysph.sph.basic_equations import IsothermalEOS
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.tools.geometry import get_2d_block
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.sph.equation import Group, Equation

from pysph.sph.surface_tension import SmoothedColor

from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

from pysph.base.nnps import DomainManager
from pysph.sph.wc.basic import TaitEOS

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

nx = 50
dx = Lx / nx
volume = dx * dx
hdx = 1.5

h0 = hdx * dx

tf = 10.0

epsilon = 0.01 / h0

dt1 = 0.25*np.sqrt(rho1*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0)

dt3 = 0.125*rho1*h0*h0/nu0

dt = 0.9*min(dt1, dt2, dt3)


class SummationDensity(Equation):

    def initialize(self, d_idx, d_V, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ, s_m, s_idx):
        d_rho[d_idx] += s_m[s_idx] * WIJ


class MomentumEquationViscosity(Equation):

    def __init__(self, dest, sources, eta=0.01):
        self.eta = eta * eta
        super(MomentumEquationViscosity, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_nu, s_nu, d_rho,
             s_rho, DWIJ, R2IJ, VIJ, HIJ, XIJ):

        dw = (DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]) / \
            (R2IJ + self.eta * HIJ * HIJ)
        mult = s_m[s_idx] * (d_nu[d_idx] + s_nu[s_idx]) / (d_rho[d_idx] * s_rho[s_idx])
        d_au[d_idx] += dw * mult * VIJ[0]
        d_av[d_idx] += dw * mult * VIJ[1]
        d_aw[d_idx] += dw * mult * VIJ[2]


class MomentumEquationPressureGradient(Equation):

    def __init__(self, dest, sources):
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_p, s_p, DWIJ, d_rho, s_rho):
        factor = -s_m[s_idx] * (d_p[d_idx] + s_p[s_idx]) / (d_rho[d_idx] * s_rho[s_idx])
        d_au[d_idx] += factor * DWIJ[0]
        d_av[d_idx] += factor * DWIJ[1]
        d_aw[d_idx] += factor * DWIJ[2]


class MorrisColorGradient(Equation):

    def __init__(self, dest, sources, epsilon=1e-6):
        self.epsilon2 = epsilon * epsilon
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
        Cba = (s_scolor[s_idx] - d_scolor[d_idx]) * s_m[s_idx] / s_rho[s_idx]

        # color gradient
        d_cx[d_idx] += Cba * DWIJ[0]
        d_cy[d_idx] += Cba * DWIJ[1]
        d_cz[d_idx] += Cba * DWIJ[2]

    def post_loop(self, d_idx, d_cx, d_cy, d_cz,
                  d_nx, d_ny, d_nz, d_N, d_ddelta):

        # # this is to avoid the very small values of
        # # normal direction which give spurious results
        # # of derivatives which build on and give unstable
        # # results at the interfaces
        # if d_cx[d_idx] * d_cx[d_idx] < self.epsilon2:
        #     d_cx[d_idx] = 0.0
        # if d_cy[d_idx] * d_cy[d_idx] < self.epsilon2:
        #     d_cy[d_idx] = 0.0
        # if d_cz[d_idx] * d_cz[d_idx] < self.epsilon2:
        #     d_cz[d_idx] = 0.0

        # absolute value of the color gradient
        mod_gradc2 = d_cx[d_idx] * d_cx[d_idx] + \
            d_cy[d_idx] * d_cy[d_idx] + \
            d_cz[d_idx] * d_cz[d_idx]

        # avoid sqrt computations on non-interface particles
        # (particles for which the color gradient is zero) Eq. (19,
        # 20) in [JM00]
        if mod_gradc2 > self.epsilon2:
            # this normal is reliable in the sense of [JM00]
            d_N[d_idx] = 1.0

            # compute the normals
            mod_gradc = 1. / sqrt(mod_gradc2)

            d_nx[d_idx] = d_cx[d_idx] * mod_gradc
            d_ny[d_idx] = d_cy[d_idx] * mod_gradc
            d_nz[d_idx] = d_cz[d_idx] * mod_gradc

            # discretized Dirac Delta function
            d_ddelta[d_idx] = 1. / mod_gradc


class FindMaxddelta(Equation):

    def reduce(self, dst, t, dt):
        max_ddelta = serial_reduce_array(dst.ddelta, 'max')
        dst.max_ddelta[0] = parallel_reduce_array(max_ddelta, 'max')


class CSFSurfaceTensionForce(Equation):

    def __init__(self, dest, sources, sigma=1.0):
        self.sigma = sigma
        super(CSFSurfaceTensionForce, self).__init__(dest, sources)

    # def initialize(self, d_idx, d_au, d_av):
    #     d_au[d_idx] = 0.0
    #     d_av[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_N, d_N, d_au, d_av, s_m, d_nx, d_ny, s_nx, s_ny, DWIJ, d_ddelta, d_rho, s_rho, s_ddelta, d_max_ddelta):
        if d_N[d_idx] == 1.0:
            d_au[d_idx] += s_m[s_idx] * (DWIJ[0] * (d_ddelta[d_idx] + s_ddelta[s_idx] - 2 * d_max_ddelta[0] - d_ddelta[d_idx] * d_nx[d_idx] * d_nx[d_idx] - s_ddelta[s_idx] * s_nx[
                s_idx] * s_nx[s_idx]) - DWIJ[1] * (d_nx[d_idx] * d_ny[d_idx] * d_ddelta[d_idx] + s_nx[s_idx] * s_ny[s_idx] * s_ddelta[s_idx])) / (d_rho[d_idx] * s_rho[s_idx])
            d_av[d_idx] += s_m[s_idx] * (-DWIJ[0] * (d_nx[d_idx] * d_ny[d_idx] * d_ddelta[d_idx] + s_nx[s_idx] * s_ny[s_idx] * s_ddelta[s_idx]) + DWIJ[1] * (
                d_ddelta[d_idx] + s_ddelta[s_idx] - 2 * d_max_ddelta[0] - d_ddelta[d_idx] * d_ny[d_idx] * d_ny[d_idx] - s_ddelta[s_idx] * s_ny[s_idx] * s_ny[s_idx])) / (d_rho[d_idx] * s_rho[s_idx])

    # def loop(self, d_idx, s_idx, d_au, d_av, s_m, d_nx, d_ny, s_nx, s_ny, DWIJ, d_ddelta, d_rho, s_rho, s_ddelta):
    #     d_au[d_idx] += s_m[s_idx]*(DWIJ[0]*(d_ddelta[d_idx]*(1.0-d_nx[d_idx]*d_nx[d_idx])+s_ddelta[s_idx]*(1.0-s_nx[s_idx]*s_nx[s_idx]))-DWIJ[1]*(d_nx[d_idx]*d_ny[d_idx]*d_ddelta[d_idx]+s_nx[s_idx]*s_ny[s_idx]*s_ddelta[s_idx]))/(d_rho[d_idx]*s_rho[s_idx])
    # d_av[d_idx] +=
    # s_m[s_idx]*(-DWIJ[0]*(d_nx[d_idx]*d_ny[d_idx]*d_ddelta[d_idx]+s_nx[s_idx]*s_ny[s_idx]*s_ddelta[s_idx])+DWIJ[1]*(d_ddelta[d_idx]*(1.0-d_ny[d_idx]*d_ny[d_idx])+s_ddelta[s_idx]*(1.0-s_ny[s_idx]*s_ny[s_idx])))/(d_rho[d_idx]*s_rho[s_idx])


class MultiPhase(Application):
    global nu0

    def create_particles(self):
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx - dx, height=Ly - dx, center=np.array([0., 0.]))
        rho_fluid = np.ones_like(fluid_x) * rho1
        m_fluid = rho_fluid * volume
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu']
        consts = {'max_ddelta': np.zeros(1, dtype=float)}
        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props, constants=consts)
        for i in range(len(fluid.x)):
            if (fluid.x[i]*fluid.x[i] + fluid.y[i]*fluid.y[i]) < 0.0625:
                fluid.color[i] = 1.0
            else:
                fluid.color[i] = 0.0
        fluid.V[:] = 1. / volume
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                 'kappa', 'N', 'scolor', 'p'])
        # angles = np.random.random_sample((len(fluid.x),))*2*np.pi
        # vel = np.sqrt(2 * KE / fluid.m)
        # fluid.u = vel*2.0*np.cos(angles)
        # fluid.v = vel*2.0*np.sin(angles)
        fluid.nu[:] = nu0
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=-0.5 * Lx, xmax=0.5 * Lx, ymin=-0.5*Ly, ymax=0.5*Ly,
            periodic_in_x=True, periodic_in_y=True)

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False, pfreq=1)
        return solver

    def create_equations(self):
        morris_equations = [
            Group(equations=[
                SummationDensity(
                    dest='fluid', sources=[
                        'fluid']),
            ], real=False, update_nnps=False),
            Group(equations=[
                # IsothermalEOS(
                #     dest='fluid',
                #     sources=None,
                #     rho0=rho1,
                #     p0=0.0,
                #     c0=c0),
                TaitEOS(dest='fluid', sources=None, rho0=rho1, c0=c0, gamma=1.0, p0=p1),
                SmoothedColor(
                    dest='fluid', sources=[
                        'fluid', ]),
                # ScaleSmoothingLength(dest='fluid', sources=None, factor=2.0/3.0),
            ], real=True, update_nnps=True),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid', ],
                                    epsilon=epsilon),
                # ScaleSmoothingLength(dest='fluid', sources=None, factor=1.5),
            ], real=False, update_nnps=False),
            # Group(equations=[
            #     InterfaceCurvatureFromNumberDensity(dest='fluid', sources=['fluid'],
            #                                         with_morris_correction=True),
            # ], real=False, update_nnps=False),
            Group(equations=[
                  FindMaxddelta(dest='fluid', sources=None),
                  ]),
            Group(equations=[
                  MomentumEquationPressureGradient(
                    dest='fluid', sources=['fluid']),
                  MomentumEquationViscosity(
                    dest='fluid', sources=['fluid']),
                  ], real=False, update_nnps=False),
            Group(
                equations=[
                    CSFSurfaceTensionForce(
                        dest='fluid', sources=['fluid']),
                ], real=False, update_nnps=False)
        ]

        return morris_equations

    def post_process(self):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load
        files = self.output_files
        ke = []
        t = []
        for f in files:
            data = load(f)
            pa = data['arrays']['fluid']
            t.append(data['solver_data']['t'])
            m = pa.m
            u = pa.u
            v = pa.v
            length = len(m)
            ke.append(sum(0.5 * m * (u**2 + v**2)))
        plt.plot(t, ke)
        fig = os.path.join(self.output_dir, "KEvst.png")
        plt.savefig(fig)
        plt.close()


if __name__ == '__main__':
    app = MultiPhase()
    app.run()
    app.post_process()
