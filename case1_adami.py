import numpy as np
import os

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
from pysph.sph.integrator import PECIntegrator, EPECIntegrator

from pysph.base.nnps import DomainManager
from math import sqrt
from surface_tension import SummationDensity, MomentumEquationPressureGradientAdami, MomentumEquationViscosityAdami, SolidWallPressureBCnoDensity, ColorGradientAdami, ConstructStressMatrix, SurfaceForceAdami

from pysph.base.nnps import DomainManager

dim = 2
Lx = 0.5
Ly = 1.0

nu = 0.0
sigma = 1.0
rho0 = 1.

c0 = 20.0
gamma = 1.4
R = 287.1

p0 = c0**2 * rho0

nx = 50
dx = Lx / nx
volume = dx * dx
hdx = 1.5

h0 = hdx * dx

tf = 0.5

epsilon = 0.01 / h0

KE = 10**(-6.6)*p0*p0*gamma/(c0 * c0 * rho0 * rho0 * nx * nx * (gamma - 1))

Vmax = np.sqrt(2 * KE / (rho0 * dx * dx))

dt = 0.9 * 0.25*h0 / (c0 + Vmax)


class MultiPhase(Application):

    def create_particles(self):
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx - dx, height=Ly - dx, center=np.array([0., 0.5 * Ly]))
        rho_fluid = np.ones_like(fluid_x) * rho0
        m_fluid = rho_fluid * volume
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu', 'dt_force', 'dt_cfl', 'pi10', 'pi11', 'pi01', 'pi00']
        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props)
        for i in range(len(fluid.x)):
            if fluid.y[i] > 0.25 and fluid.y[i] < 0.75:
                fluid.color[i] = 1.0
            else:
                fluid.color[i] = 0.0
        fluid.V[:] = 1. / volume
        fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta',
                                 'kappa', 'N', 'scolor', 'p'])
        angles = np.random.random_sample((len(fluid.x),))*2*np.pi
        vel = np.sqrt(2 * KE / fluid.m)
        fluid.u = vel  # *2.0*np.cos(angles)
        fluid.v = vel  # *2.0*np.sin(angles)
        fluid.nu[:] = 0.0
        return [fluid]

    def create_domain(self):
        return DomainManager(
            xmin=-0.5 * Lx, xmax=0.5 * Lx, ymin=0.0, ymax=Ly,
            periodic_in_x=True, periodic_in_y=True, n_layers=6)

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = EPECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False)
        return solver

    def create_equations(self):
        adami_equations = [
            Group(equations=[
                SummationDensity(
                    dest='fluid', sources=['fluid']),
            ], real=False),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None,
                        rho0=rho0, c0=c0, gamma=1, p0=p0),
            ], real=False),
            Group(equations=[
                ColorGradientAdami(dest='fluid', sources=['fluid',]),
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
        return adami_equations


if __name__ == '__main__':
    app = MultiPhase()
    app.run()
