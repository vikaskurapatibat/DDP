from pysph.sph.wc.basic import MomentumEquation
import numpy as np
import os

from math import sqrt

from pysph.solver.utils import load
from pysph.sph.wc.basic import TaitEOS

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


from surface_tension import SolidWallPressureBCnoDensity, SummationDensitySourceMass, MomentumEquationViscosityMorris, MorrisColorGradient, InterfaceCurvatureFromDensity, MomentumEquationPressureGradientMorris


dim = 2

Lx = 1.0
Ly = 1.0

nu0 = 0.05
sigma = 1.0
rho0 = 1.0
c0 = 20.0
gamma = 1.0
p0 = c0*c0*rho0
nx = 120
dx = Lx/nx
hdx = 1.5
h0 = hdx*dx
tf = 0.5
epsilon = 0.01/h0
volume = dx*dx
r0 = 0.05
v0 = 10.0

dt1 = 0.25*np.sqrt(rho0*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0)

dt3 = 0.125*rho0*h0*h0/nu0

dt = 0.9*min(dt1, dt2, dt3)


def r(x, y):
    return x*x + y*y


class MultiPhase(Application):
    global nu0

    def create_particles(self):
        fluid_x, fluid_y = get_2d_block(
            dx=dx, length=Lx, height=Ly, center=np.array([0., 0.]))
        rho_fluid = np.ones_like(fluid_x) * rho0
        m_fluid = rho_fluid * volume
        h_fluid = np.ones_like(fluid_x) * h0
        cs_fluid = np.ones_like(fluid_x) * c0
        wall_x, wall_y = get_2d_block(dx=dx, length=Lx+6*dx, height=Ly+6*dx, center = np.array([0., 0.]))
        rho_wall = np.ones_like(wall_x) * rho0
        m_wall = rho_wall * volume
        h_wall = np.ones_like(wall_x) * h0
        cs_wall = np.ones_like(wall_x) * c0
        additional_props = ['V', 'color', 'scolor', 'cx', 'cy', 'cz',
                            'cx2', 'cy2', 'cz2', 'nx', 'ny', 'nz', 'ddelta',
                            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat',
                            'ax', 'ay', 'az', 'wij', 'vmag2', 'N', 'wij_sum',
                            'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0',
                            'kappa', 'arho', 'nu', 'wg', 'ug', 'vg',
                            'pi00', 'pi01', 'pi10', 'pi11', 'dt_force', 'dt_cfl']
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
            dt=dt, tf=tf, adaptive_timestep=False,
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
                TaitEOS(dest='fluid', sources=None, rho0=rho0, c0=c0, gamma=1, p0=0.0),
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
                    # MomentumEquation(dest='fluid', sources=['fluid', 'wall'], c0=c0, tensile_correction=False, alpha=0.0, beta=0.0),
                    MomentumEquationPressureGradientMorris(dest='fluid', sources=['fluid', 'wall']),
                    MomentumEquationViscosityMorris(dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForce(
                        dest='fluid', sources=None, sigma=sigma),
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu0)
                ]),
                ]
        return morris_equations

if __name__=='__main__':
    app = MultiPhase()
    app.run()
