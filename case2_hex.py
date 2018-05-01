import numpy as np
import os

from pysph.sph.basic_equations import IsothermalEOS
from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

from pysph.tools.geometry import get_2d_block
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, QuinticSpline
from pysph.sph.equation import Group, Equation

from pysph.sph.surface_tension import SmoothedColor, CSFSurfaceTensionForce

from pysph.solver.application import Application
from pysph.solver.solver import Solver

from pysph.sph.integrator_step import TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

from pysph.base.nnps import DomainManager
from pysph.sph.wc.basic import TaitEOS
from surface_tension import SummationDensitySourceMass, MomentumEquationViscosityMorris, MomentumEquationPressureGradientMorris, InterfaceCurvatureFromDensity, MorrisColorGradient

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
p1 = c0*c0*rho1
nx = 50
dx = Lx / nx
volume = dx * dx
hdx = 1.5

h0 = hdx*dx

tf = 10.0

epsilon = 0.01 / h0

dt1 = 0.25*np.sqrt(rho1*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0)

dt3 = 0.125*rho1*h0*h0/nu0

dt = 0.9*min(dt1, dt2, dt3)


def radius(x, y):
    return x*x + y*y


class MultiPhase(Application):
    global nu0

    def create_particles(self):
        x, y = np.mgrid[-0.5*Lx:0.5*Lx:dx, -0.5*Ly:0.5*Ly:dx]
        xc = x.copy() + 0.5*dx
        yc = y.copy() + 0.5*dx
        fluid_x = np.concatenate([x.ravel(), xc.ravel()]) + 0.25*dx
        fluid_y = np.concatenate([y.ravel(), yc.ravel()]) + 0.25*dx
        rho_fluid = np.ones_like(fluid_x) * rho1
        m_fluid = rho_fluid*volume*0.5
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu']
        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props)
        for i in range(len(fluid.x)):
            if (fluid.x[i] * fluid.x[i] + fluid.y[i] * fluid.y[i]) < 0.0625:
                fluid.color[i] = 1.0
            else:
                fluid.color[i] = 0.0
        fluid.V[:] = 1. / volume
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                 'kappa', 'N', 'scolor', 'p'])
        fluid.nu[:] = nu0
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=-0.5 * Lx, xmax=0.5 * Lx, ymin=-0.5 * Ly, ymax=0.5 * Ly,
            periodic_in_x=True, periodic_in_y=True)

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        morris_equations = [
            Group(equations=[
                SummationDensitySourceMass(
                    dest='fluid', sources=[
                        'fluid']),
            ], real=False, update_nnps=False),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None,
                        rho0=rho1, c0=c0, gamma=1.0),
                SmoothedColor(
                    dest='fluid', sources=[
                        'fluid', ]),
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

        return morris_equations

    def post_process(self):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load
        files = self.output_files
        dp = []
        t = []
        for f in files:
            data = load(f)
            pa = data['arrays']['fluid']
            t.append(data['solver_data']['t'])
            m = pa.m
            x = pa.x
            y = pa.y
            N = pa.N
            p = pa.p
            n = len(m)
            count_in = 0
            count_out = 0
            p_in = 0
            p_out = 0

            for i in range(n):
                r = radius(x[i], y[i])
                if N[i] < 1:
                    if radius(x[i], y[i]) < 0.0625:
                        p_in += p[i]
                        count_in += 1
                    else:
                        p_out += p[i]
                        count_out += 1
                else:
                    continue
            dp.append((p_in/count_in) - (p_out/count_out))

        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, dp=dp)
        plt.plot(t, dp)
        fig = os.path.join(self.output_dir, "dpvst.png")
        plt.savefig(fig)
        plt.close()


if __name__ == '__main__':
    app = MultiPhase()
    app.run()
    app.post_process()
