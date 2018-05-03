from pysph.sph.equation import Group, Equation


class SurfaceForceAdami(Equation):
    def initialize(self, d_au, d_av, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0

    def loop(self, d_au, d_av, d_idx, d_m, DWIJ, d_pi00, d_pi01, d_pi10, d_pi11, s_pi00, s_pi01, s_pi10, s_pi11, d_V, s_V, s_idx):
        f1 = (d_pi00[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi00[s_idx]/(s_V[s_idx]*s_V[s_idx]))
        f2 = (d_pi01[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi01[s_idx]/(s_V[s_idx]*s_V[s_idx]))
        f3 = (d_pi10[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi10[s_idx]/(s_V[s_idx]*s_V[s_idx]))
        f4 = (d_pi11[d_idx]/(d_V[d_idx]*d_V[d_idx]) + s_pi11[s_idx]/(s_V[s_idx]*s_V[s_idx]))
        d_au[d_idx] += (DWIJ[0]*f1 + DWIJ[1]*f2)/d_m[d_idx]
        d_av[d_idx] += (DWIJ[0]*f3 + DWIJ[1]*f4)/d_m[d_idx]


class ConstructStressMatrix(Equation):

    def __init__(self, dest, sources, sigma, d=2):
        self.sigma = sigma
        self.d = d
        super(ConstructStressMatrix, self).__init__(dest, sources)

    def initialize(self, d_pi00, d_pi01, d_pi10, d_pi11, d_cx, d_cy, d_idx):
        mod_gradc2 = d_cx[d_idx]*d_cx[d_idx] + d_cy[d_idx]*d_cy[d_idx]
        mod_gradc = sqrt(mod_gradc2)
        if mod_gradc > 1e-14:
            factor = self.sigma/mod_gradc
            d_pi00[d_idx] = (-d_cx[d_idx]*d_cx[d_idx] + (mod_gradc2)/self.d)*factor
            d_pi01[d_idx] = -factor*d_cx[d_idx]*d_cy[d_idx]
            d_pi10[d_idx] = -factor*d_cx[d_idx]*d_cy[d_idx]
            d_pi11[d_idx] = (-d_cy[d_idx]*d_cy[d_idx] + (mod_gradc2)/self.d)*factor
        else:
            d_pi00[d_idx] = 0.0
            d_pi01[d_idx] = 0.0
            d_pi10[d_idx] = 0.0
            d_pi11[d_idx] = 0.0


class ColorGradientAdami(Equation):

    def initialize(self, d_idx, d_cx, d_cy, d_cz):
        d_cx[d_idx] = 0.0
        d_cy[d_idx] = 0.0
        d_cz[d_idx] = 0.0

    def loop(self, d_idx, d_cx, d_cy, d_cz, d_V, s_V, d_color, s_color, DWIJ, s_idx):
        c_i = d_color[d_idx]/(d_V[d_idx]*d_V[d_idx])
        c_j = s_color[s_idx]/(s_V[s_idx]*s_V[s_idx])
        factor = d_V[d_idx]*(c_i + c_j)
        d_cx[d_idx] += factor*DWIJ[0]
        d_cy[d_idx] += factor*DWIJ[1]
        d_cz[d_idx] += factor*DWIJ[2]


class MomentumEquationViscosityAdami(Equation):

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx, d_m, R2IJ, XIJ, EPS, VIJ, d_nu, s_nu):
        factor = 2.0*d_nu[d_idx]*s_nu[s_idx]/(d_nu[d_idx] + s_nu[s_idx])
        V_i = 1/(d_V[d_idx]*d_V[d_idx])
        V_j = 1/(s_V[s_idx]*s_V[s_idx])
        dwijdotrij = (DWIJ[0]*XIJ[0]+DWIJ[1]*XIJ[1]+DWIJ[2]*XIJ[2])
        dwijdotrij /= (R2IJ + EPS)
        factor = factor*(V_i+V_j)*dwijdotrij/d_m[d_idx]
        d_au[d_idx] += factor*VIJ[0]
        d_av[d_idx] += factor*VIJ[1]
        d_aw[d_idx] += factor*VIJ[2]


class MomentumEquationPressureGradientAdami(Equation):

    def initialize(self, d_au, d_av, d_aw, d_idx):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_au, d_av, d_aw, s_V, d_p, s_p, DWIJ, s_idx, d_m):
        p_i = d_p[d_idx]/(d_V[d_idx]*d_V[d_idx])
        p_j = s_p[s_idx]/(s_V[s_idx]*s_V[s_idx])
        d_au[d_idx] += -(p_i+p_j)*DWIJ[0]/d_m[d_idx]
        d_av[d_idx] += -(p_i+p_j)*DWIJ[1]/d_m[d_idx]
        d_aw[d_idx] += -(p_i+p_j)*DWIJ[2]/d_m[d_idx]


class SummationDensity(Equation):

    def initialize(self, d_idx, d_V, d_rho):
        d_rho[d_idx] = 0.0
        d_V[d_idx] = 0.0

    def loop(self, d_idx, d_V, d_rho, d_m, WIJ, s_idx):
        d_rho[d_idx] += d_m[d_idx]*WIJ
        d_V[d_idx] += WIJ


class MomentumEquationViscosityMorris(Equation):

    def __init__(self, dest, sources, eta=0.01):
        self.eta = eta*eta
        super(MomentumEquationViscosityMorris, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_nu, s_nu, d_rho,
             s_rho, DWIJ, R2IJ, VIJ, HIJ, XIJ):

        dw = (DWIJ[0]*XIJ[0]+DWIJ[1]*XIJ[1]+DWIJ[2]*XIJ[2])/(R2IJ+self.eta*HIJ*HIJ)
        mult = s_m[s_idx]*(d_nu[d_idx]+s_nu[s_idx])/(d_rho[d_idx]*s_rho[s_idx])
        d_au[d_idx] += dw*mult*VIJ[0]
        d_av[d_idx] += dw*mult*VIJ[1]
        d_aw[d_idx] += dw*mult*VIJ[2]


class MomentumEquationPressureGradientMorris(Equation):

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_au, d_av, d_aw, s_m, d_p, s_p, DWIJ, d_rho, s_rho):
        factor = -s_m[s_idx]*(d_p[d_idx] + s_p[s_idx])/(d_rho[d_idx]*s_rho[s_idx])
        d_au[d_idx] += factor*DWIJ[0]
        d_av[d_idx] += factor*DWIJ[1]
        d_aw[d_idx] += factor*DWIJ[2]


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


class FindMaxddelta(Equation):

    def reduce(self, dst, t, dt):
        max_ddelta = serial_reduce_array(dst.ddelta, 'max')
        dst.max_ddelta[0] = parallel_reduce_array(max_ddelta, 'max')


class SurfaceTensionForce(Equation):

    def __init__(self, dest, sources, sigma=1.0):
        self.sigma = sigma
        super(SurfaceTensionForce, self).__init__(dest, sources)

    # def initialize(self, d_idx, d_au, d_av):
    #     d_au[d_idx] = 0.0
    #     d_av[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_N, d_N, d_au, d_av, s_m, d_nx, d_ny, s_nx, s_ny, DWIJ, d_ddelta, d_rho, s_rho, s_ddelta, d_max_ddelta):
        if d_N[d_idx] == 1.0:
            d_au[d_idx] += s_m[s_idx] * (DWIJ[0] * (d_ddelta[d_idx] + s_ddelta[s_idx] - 2 * d_max_ddelta[0] - d_ddelta[d_idx] * d_nx[d_idx] * d_nx[d_idx] - s_ddelta[s_idx] * s_nx[
                s_idx] * s_nx[s_idx]) - DWIJ[1] * (d_nx[d_idx] * d_ny[d_idx] * d_ddelta[d_idx] + s_nx[s_idx] * s_ny[s_idx] * s_ddelta[s_idx])) / (d_rho[d_idx] * s_rho[s_idx])
            d_av[d_idx] += s_m[s_idx] * (-DWIJ[0] * (d_nx[d_idx] * d_ny[d_idx] * d_ddelta[d_idx] + s_nx[s_idx] * s_ny[s_idx] * s_ddelta[s_idx]) + DWIJ[1] * (
                d_ddelta[d_idx] + s_ddelta[s_idx] - 2 * d_max_ddelta[0] - d_ddelta[d_idx] * d_ny[d_idx] * d_ny[d_idx] - s_ddelta[s_idx] * s_ny[s_idx] * s_ny[s_idx])) / (d_rho[d_idx] * s_rho[s_idx])


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
        if d_wij_sum[d_idx] > 1e-12:
            d_kappa[d_idx] /= d_wij_sum[d_idx]
        d_kappa[d_idx] *= -self.dim


class CSFSurfaceTensionForceAdami(Equation):
    def initialize(self, d_idx, d_au, d_av, d_aw, d_kappa, d_cx, d_cy, d_cz, d_m, d_alpha, d_rho):
        d_au[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cx[d_idx]/d_rho[d_idx]
        d_av[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cy[d_idx]/d_rho[d_idx]
        d_aw[d_idx] += -d_alpha[d_idx]*d_kappa[d_idx]*d_cz[d_idx]/d_rho[d_idx]
