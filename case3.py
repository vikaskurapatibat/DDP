
import numpy as np
import os

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
from pysph.sph.integrator import PECIntegrator

from surface_tension import SolidWallPressureBCnoDensity, SummationDensitySourceMass, MomentumEquationViscosityMorris, MomentumEquationPressureGradientMorris, MorrisColorGradient, InterfaceCurvatureFromDensity

dim = 2
Lx = 1.0
Ly = 1.0

nu = 0.05
sigma = 2.0
factor1 = 0.5
factor2 = 1 / factor1
rho0 = 1.0

c0 = 30.0
gamma = 1.4
R = 287.1

p0 = c0**2 * rho0

nx = 120
dx = Lx / nx
volume = dx * dx
hdx = 1.5

h0 = hdx * dx

tf = 0.5

epsilon = 0.01 / h0

r0 = 0.05
v0 = 10.0


dt1 = 0.25*np.sqrt(rho0*h0*h0*h0/(2.0*np.pi*sigma))

dt2 = 0.25*h0/(c0)

dt3 = 0.125*rho0*h0*h0/nu

dt = 0.9*min(dt1, dt2, dt3)


def r(x, y):
    return x*x + y*y


class MultiPhase(Application):

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
                            'pi00', 'pi01', 'pi10', 'pi11', 'alpha']
        consts = {'max_ddelta': np.zeros(1, dtype=float)}

        fluid = get_particle_array(
            name='fluid', x=fluid_x, y=fluid_y, h=h_fluid, m=m_fluid,
            rho=rho_fluid, cs=cs_fluid, additional_props=additional_props, constants=consts)
        for i in range(len(fluid.x)):
            if (fluid.x[i]*fluid.x[i] + fluid.y[i]*fluid.y[i]) < 0.1875*0.1875:
                fluid.color[i] = 1.0
            else:
                fluid.color[i] = 0.0
        fluid.alpha[:] = sigma
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
        fluid.nu[:] = nu

        return [fluid, wall]

    def create_solver(self):
        kernel = QuinticSpline(dim=2)
        integrator = PECIntegrator(fluid=TransportVelocityStep())
        solver = Solver(
            kernel=kernel, dim=dim, integrator=integrator,
            dt=dt, tf=tf, adaptive_timestep=False,
            output_at_times=[0., 0.08, 0.16, 0.26])
        return solver

    def add_user_options(self, group):
        choices = ['morris', 'tvf', 'adami_stress', 'adami', 'shadloo']
        group.add_argument(
            "--scheme", action="store", dest='scheme', default='morris',
            choices=choices,
            help='Specify scheme to use among %s' % choices
        )

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
                TaitEOS(dest='fluid', sources=None, rho0=rho0, c0=c0, gamma=7, p0=0.0),
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
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu)
                ]),
                ]

        sy11_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'wall']),
                SummationDensity(dest='wall', sources=['fluid', 'wall'])
            ], ),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SolidWallPressureBCnoDensity(dest='wall', sources=['fluid']),
                SY11ColorGradient(dest='fluid', sources=['fluid', 'wall'])
            ], ),
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor1),
            ], update_nnps=True),
            Group(equations=[
                SY11DiracDelta(dest='fluid', sources=['fluid', 'wall'])
            ],
            ),
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid', 'wall'],
                    with_morris_correction=True),
            ], ),
            Group(equations=[
                ScaleSmoothingLength(dest='fluid', sources=None,
                                     factor=factor2),
            ], update_nnps=True,
            ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'wall'], pb=p0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),
                    ShadlooYildizSurfaceTensionForce(dest='fluid',
                                                     sources=None,
                                                     sigma=sigma),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid']),
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu)
                ], )
        ]

        adami_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'wall']),
                SummationDensity(dest='wall', sources=['fluid', 'wall'])
            ],),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SolidWallPressureBCnoDensity(dest='wall', sources=['fluid']),
            ],),
            Group(equations=[
                AdamiColorGradient(dest='fluid', sources=['fluid', 'wall']),
            ],
            ),
            Group(equations=[
                AdamiReproducingDivergence(dest='fluid', sources=['fluid', 'wall'],
                                           dim=2),
            ],),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'wall'], pb=p0),
                    MomentumEquationViscosityAdami(
                        dest='fluid', sources=['fluid']),
                    CSFSurfaceTensionForceAdami(dest='fluid', sources=None,),
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu),
                ], )
        ]

        adami_stress_equations = [
            Group(equations=[
                SummationDensity(
                    dest='fluid', sources=[
                        'fluid', 'wall']),
                SummationDensity(
                    dest='wall', sources=[
                        'fluid', 'wall']),
            ]),
            Group(equations=[
                TaitEOS(dest='fluid', sources=None,
                        rho0=rho0, c0=c0, gamma=7, p0=p0),
                SolidWallPressureBCnoDensity(dest='wall', sources=['fluid']),
            ]),
            Group(equations=[
                ColorGradientAdami(dest='fluid', sources=['fluid', 'wall']),
            ]),
            Group(equations=[ConstructStressMatrix(
                dest='fluid', sources=None, sigma=sigma, d=2)]),
            Group(
                equations=[
                    MomentumEquationPressureGradientAdami(
                        dest='fluid', sources=['fluid', 'wall']),
                    MomentumEquationViscosityAdami(
                        dest='fluid', sources=['fluid']),
                    SurfaceForceAdami(
                        dest='fluid', sources=['fluid', 'wall']),
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu)
                ]),
        ]

        tvf_equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'wall']),
                SummationDensity(dest='wall', sources=['fluid', 'wall'])
            ], ),
            Group(equations=[
                StateEquation(dest='fluid', sources=None, rho0=rho0,
                              p0=p0),
                SolidWallPressureBCnoDensity(dest='wall', sources=['fluid']),
                SmoothedColor(dest='fluid', sources=['fluid', 'wall']),
                SmoothedColor(dest='wall', sources=['fluid', 'wall']),                
            ], ),
            Group(equations=[
                MorrisColorGradient(dest='fluid', sources=['fluid', 'wall'],
                                    epsilon=epsilon),
            ], 
            ),
            Group(equations=[
                InterfaceCurvatureFromNumberDensity(
                    dest='fluid', sources=['fluid', 'wall'],
                    with_morris_correction=True),
            ], ),
            Group(
                equations=[
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['fluid', 'wall'], pb=p0),
                    MomentumEquationViscosity(
                        dest='fluid', sources=['fluid'], nu=nu),
                    CSFSurfaceTensionForce(dest='fluid', sources=None,
                                           sigma=sigma),
                    MomentumEquationArtificialStress(dest='fluid',
                                                     sources=['fluid']),
                    SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu)
                ], )
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

    def post_process(self):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load
        files = self.output_files
        amat = []
        t = []
        centerx = []
        centery = []
        velx = []
        vely = []

        for f in files:
            data = load(f)
            pa = data['arrays']['fluid']
            t.append(data['solver_data']['t'])
            x = pa.x
            y = pa.y
            u = pa.u
            v = pa.v
            color = pa.color
            length = len(color)
            min_x = 0.0
            max_x = 0.0
            cx = 0
            cy = 0
            vx = 0
            vy = 0
            for i in range(length):
                if color[i] == 1:
                    if x[i] < min_x:
                        min_x = x[i]
                    if x[i] > max_x:
                        max_x = x[i]
                    else:
                        continue
                    if x[i] > 0 and y[i] > 0:
                        cx += x[i]
                        cy += y[i]
                        vx += u[i]
                        vy += v[i]
                        count += 1
                    else:
                        continue

                else:
                    continue
            amat.append(0.5*(max_x - min_x))
            centerx.append(cx/count)
            centery.append(cy/count)
            velx.append(vx/count)
            vely.append(vy/count)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, semimajor=amat, centerx=centerx, centery=centery,
                 velx=velx, vely=vely)
        plt.plot(t, amat)
        fig = os.path.join(self.output_dir, 'semimajorvst.png')
        plt.savefig(fig)
        plt.close()
        plt.plot(t, centerx, label='x position')
        plt.plot(t, centery, label='y position')
        plt.legend()
        fig1 = os.path.join(self.output_dir, 'centerofmassposvst')
        plt.savefig(fig1)
        plt.close()
        plt.plot(t, velx, label='x velocity')
        plt.plot(t, vely, label='y velocity')
        plt.legend()
        fig2 = os.path.join(self.output_dir, 'centerofmassvelvst')
        plt.savefig(fig2)
        plt.close()


if __name__ == '__main__':
    app = MultiPhase()
    app.run()
    app.post_process()

# Generalised TVF 2017 jcp Hu and Adams
