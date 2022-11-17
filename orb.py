import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as anime

import os, sys

SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN = 0, 1, 2, 3, 4, 5, 6
TS, XS, YS, ZS, XRANGE, YRANGE, ZRANGE = 0, 1, 2, 3, 4, 5, 6

atol = 1e-6
rtol = 1e-8

fig_index = 0

#################################################################################################

au2m = 1.495978707e11
m2au = 1 / au2m 

yr2sec = 365 * 24 * 60 * 60
sec2yr = 1 / yr2sec

names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn"]

N = len(names)
N3 = N * 3

_masses = np.array([1.98844e30, 3.30104e23, 4.86732e24, 5.97219e24, 6.41693e23, 1.89813e27, 5.68319e26])
masses = _masses / _masses[EARTH]

majors = np.array([0, 0.387098261, 0.723333771, 1, 1.523706365, 5.20287342, 9.536651047])
_majors = au2m * majors

eccentricities = np.array([0, 0.20563593, 0.00677672, 0.01671123, 0.0933941, 0.04838624, 0.05386179])

periods = np.array([0, 0.2408425, 0.6151866, 1.0, 1.8808149, 11.862409, 29.446986])
_periods = yr2sec * periods

perihelion = majors * (1 - eccentricities)
_perihelion = au2m * perihelion

total_mass = np.sum(masses)

_G = 6.6743015e-11
G = 4 * np.pi ** 2 / total_mass

velocity = np.sqrt(np.divide(
    G * total_mass * (1 + eccentricities) / (1 - eccentricities), 
    majors, 
    out=np.zeros_like(majors), 
    where=majors!=0
)) # in AU / yr

inclination = np.array([0, 7.01, 3.39, 0, 1.85, 1.31, 2.49]) / 180 * np.pi # 0.77 1.77 17.14

arguments = {
    "Mass": masses,
    "Major": majors,
    "Eccentricity": eccentricities,
    "Period": periods,
    "Perihelion": perihelion,
    "Inclination": inclination,

    "Include": [0, 1],
    "Method": "Newton"
}

init_position = perihelion * np.concatenate([np.cos(inclination), np.zeros(N), np.sin(inclination)]).reshape(3, N)
init_position[0, SUN] = init_position[1, SUN] = init_position[2, SUN]
init_position = init_position.T

init_velocity = np.concatenate([np.zeros(N), velocity, np.zeros(N)]).reshape(3, N).T


###################################################################################################

def accelaration_newton(pos_this: np.ndarray, pos_that: np.ndarray, mass_that: float, G: float):
    r = pos_that - pos_this   # type: ignore
    res = G * mass_that * r / np.linalg.norm(r) ** 3
    return res

def advance_newton(t: float, state: np.ndarray, args: dict):
    # x * N3 + v * N3
    pos = state[:N3].reshape(N, 3)
    vel = state[N3:].reshape(N, 3)

    mass, major, ecc, period, peri, incl, inc, _ = args.values()
    if not len(inc):
        inc = list(range(N))
    for i in range(N):
        if i not in inc:
            vel[i] = np.zeros(3)

    step_pos = vel.copy()
    step_vec = np.zeros((N, 3))

    for i in inc: # for the i-th planet
        acc = np.zeros(3)
        for j in inc: 
            if j == i: 
                continue
            acc += accelaration_newton(pos[i, :], pos[j, :], mass[j], G)
        step_vec[i, :] = acc

    res = np.concatenate((step_pos.flatten(), step_vec.flatten()))
    return res

def adapter(t: float, state: np.ndarray, args: dict, method: str):
    return advance_newton(t, state, args)

def get_components(solution):
    ts = solution.t
    ys = solution.y.T

    poses = np.array([x[:N3].reshape(N, 3) for x in ys])

    xs = np.array([poses[:, i, 0] for i in range(N)]).T
    ys = np.array([poses[:, i, 1] for i in range(N)]).T
    zs = np.array([poses[:, i, 2] for i in range(N)]).T

    xrange = (min(xs.flatten()) - 1, max(xs.flatten()) + 1)
    yrange = (min(ys.flatten()) - 1, max(ys.flatten()) + 1)
    zrange = (min(zs.flatten()) - 1, max(zs.flatten()) + 1)

    return (ts, xs, ys, zs, xrange, yrange, zrange)

def draw_orbits(components, incs):
    if not len(incs):
        incs = list(range(N))

    xs_copy = components[XS][:, incs].copy()
    ys_copy = components[YS][:, incs].copy()
    zs_copy = components[ZS][:, incs].copy()

    plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['figure.dpi'] = 80  
    plt.ioff()

    fig = plt.figure(fig_index)
    ax = fig.axes
    ax = plt.axes(projection='3d')

    plt.xlabel("X")
    plt.ylabel("Y")

    for i in range(len(xs_copy[0, :])):
        ax.plot3D(xs_copy[:, i], ys_copy[:, i], zs_copy[:, i])  # type: ignore
    plt.xlim(components[XRANGE][0], components[XRANGE][1])
    plt.ylim(components[YRANGE][0], components[YRANGE][1])
    ax.set_zlim(components[ZRANGE][0], components[ZRANGE][1])  # type: ignore

    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z") # type: ignore

    plt.legend(["Curve", "X, Y", "X, Z", "Y, Z"])

    plt.show()

def draw_orbits_anime(ts, xs, ys, zs, xlim, ylim, zlim):
    pass

###################################################################################################

t_span = np.array([0, 50])
y0 = np.concatenate((init_position.flatten(), init_velocity.flatten()))

sol = solve_ivp(lambda t, state: adapter(t, state, arguments, "Newton"), t_span, y0, atol=atol, rtol=rtol)
sol = get_components(sol)

draw_orbits(sol, [0, 1])

sys.pause(1)
