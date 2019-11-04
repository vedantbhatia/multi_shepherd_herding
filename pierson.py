import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

scale = 20
drivers = None
agents = None
num_sheep = 40
num_drivers = 14
l = 12.5
r_nought = 15
radius = 15
k = -0.01
kd = 1
dt = 1
sheep_footprint = 5


class sheep:

    def __init__(self):
        self.coords = [np.random.uniform(0.75 - sheep_footprint/scale, 0.75 + sheep_footprint/scale, 2).reshape(2) * scale for i in range(num_sheep)]
        print(self.coords)
        self.history = [[] for i in range(num_sheep)]

    def mean_coords(self):
        sx = [i[0] for i in self.coords]
        sy = [i[1] for i in self.coords]
        mx = sum(sx) / num_sheep
        my = sum(sy) / num_sheep
        return (format([mx, my]))

    def ideal_phi(self):
        mc = self.mean_coords()
        radians = math.atan2(mc[1], mc[0]) + np.pi
        degrees = math.degrees(radians)
        return (radians, degrees)

    def qx(self):
        rads, _ = self.ideal_phi()
        p_x = math.cos(rads)
        p_y = math.sin(rads)
        return (format([p_x, p_y]))

    def qy(self):
        x = self.qx()
        return (format([-x[1], x[0]]))

    def offset(self):
        mc = self.mean_coords()
        p = np.add(mc, l * self.qx())
        return (p)

    def ideal_velocity(self, p_dot):
        qx = self.qx()
        return (format([p_dot[0] * qx[0], p_dot[1] * qx[1]]))

    def sheep_repulsion(self, i):
        s_i = self.coords[i]
        s_dot_i = 0
        for j in range(num_drivers):
            numr = -np.add(drivers.coords[j], -s_i)
            denom = magnitude(numr) ** 3
            s_dot_i += numr / denom

        return (s_dot_i)

    def radial_controller(self):
        global radius
        r_dot = (r_nought - radius)

        s_bar = self.mean_coords()
        s_i_dots = [self.sheep_repulsion(i) for i in range(num_sheep)]
        s_bar_dot = sum(s_i_dots) / num_sheep

        for i in range(num_sheep):
            s_i = self.coords[i]

            a = (s_i - s_bar)
            b = (s_i_dots[i] - s_bar_dot)
            c = a[0] * b[0] + a[1] * b[1]
            r_dot += c / num_drivers

        return (r_dot)

    def timestep(self):

        for i in range(num_sheep):
            self.history[i].append(np.copy(self.coords[i]))
            self.coords[i] = np.add(self.coords[i], dt * self.sheep_repulsion(i))


class shepherd:

    def __init__(self):
        self.coords = [np.random.rand(2) * -scale for i in range(num_drivers)]
        self.theta = [0] * num_drivers
        self.history = [[] for i in range(num_drivers)]

    def ideal_positions(self, delta_js, phi_star):
        ideal_pos = []
        s = agents.mean_coords()
        for i in range(num_drivers):
            angle = delta_js[i] + phi_star + np.pi
            d_j_star = s + radius * format([math.cos(angle), math.sin(angle)])
            ideal_pos.append(d_j_star)

        return (ideal_pos)

    def tracking_controllers(self, d_j_stars):
        d_dot_js = []
        for i in range(num_drivers):
            d_dot_js.append(-kd * (np.add(d_j_stars[i], - self.coords[i])))

        return (d_dot_js)

    def timestep(self, d_dot_js):
        for i in range(num_drivers):
            self.history[i].append(np.copy(self.coords[i]))
            self.coords[i] = np.add(self.coords[i], -dt * d_dot_js[i])


def reset():
    global agents
    global drivers
    agents = sheep()
    drivers = shepherd()


def check_radius_initialization():
    C = radius*(math.sqrt((1-math.cos(2*np.pi/num_drivers))/2)) # C/2 in paper
    r1 = radius*math.cos(np.pi/num_drivers)
    assert r1 - sheep_footprint < C, "radius = {0}, rs = {1}, C/2 = {2}".format(r1,sheep_footprint,C)
    assert sheep_footprint<l<radius, "footprint, offset, radius initialization"

def format(a):
    a = np.array(a).reshape(2)
    return (a)


def normalize(v):
    denom = (v[0, 0] ** 2 + v[0, 1] ** 2) ** 0.5
    if (denom == 0):
        return (v)
    return (v / denom)


def magnitude(v):
    return (math.sqrt(v[0] ** 2 + v[1] ** 2))


def sinfx(d):
    m = num_drivers
    return (np.sin((m * d) / (2 - 2 * m)) / np.sin(d / (2 - 2 * m)))


def delta_given_v(v):
    i2 = 10000
    search_space = np.linspace(0, 2 * np.pi, i2)
    i1 = 0
    target = magnitude(v) * radius ** 2  ## must be at most equal to num_drivers for solution to exist
    assert target <= num_drivers, "velocity - {0}, radius = {1}, target = {2} ".format(magnitude(v),radius,target)
    mn = np.ones((1, 2)) * np.inf
    while (i1 < i2):
        mid = (i2 + i1) // 2
        if (mid == i1 or mid == i2):
            break
        curr = sinfx(search_space[mid])
        if (np.abs(curr - target) < np.abs(mn[0, 0] - target)):
            mn[0, 0] = curr
            mn[0, 1] = search_space[mid]
        if (curr < target):
            i2 = mid
        else:
            i1 = mid
    return (format(mn)[1])


def distribute_deltas(delta):
    delta_js = []

    for i in range(1, num_drivers + 1):
        m = (2 * i - num_drivers - 1) / (2 * num_drivers - 2)
        d_i = delta * m
        delta_js.append(d_i)

    return (delta_js)


def herding_control():
    global radius

    # step 1: controller for p_dot (15)
    p_dot = -k * agents.offset()

    # step 2: ideal heading phi_star and ideal velocity v_star (3)
    phi_star, phi_star_deg = agents.ideal_phi()
    v_star = agents.ideal_velocity(p_dot)

    # step 3: ideal delta stars for each dog (11)
    delta = delta_given_v(v_star)
    delta_js = distribute_deltas(delta)

    # step 4: desired dog positions (16)
    d_j_stars = drivers.ideal_positions(delta_js, phi_star)

    # step 5: calculate radial controller for r_dot (14)
    r_dot = agents.radial_controller()
    print(r_dot,"radial controller")

    # step 6: calculate tracking controller (17)
    d_dot_js = drivers.tracking_controllers(d_j_stars)

    # change position of sheep, position of dogs, and radius according to controllers
    drivers.timestep(d_dot_js)
    agents.timestep()
    radius += r_dot * dt
    print(radius,"radius")
    plot_current(save=True)


def plot_current(save=False):
    plt.xlim([-2 * scale, 2 * scale])
    plt.ylim([-2 * scale, 2 * scale])
    offset = agents.offset()
    plt.plot(offset[0], offset[1], marker='x', markersize=3, color='orange')
    a = np.array(agents.history).reshape(num_sheep, -1, 2)
    for i in range(num_sheep):
        # plt.plot(agents.history[i, :, 0], agents.history[i, :, 1], '--', color="red")
        plt.plot(agents.coords[i][0], agents.coords[i][1], marker='o', markersize=3, color="red")

    d = np.array(drivers.history).reshape(num_drivers, -1, 2)
    for i in range(num_drivers):
        # plt.plot(d.history[i, :, 0], d.history[i, :, 1], ':', color="green")
        plt.plot(drivers.coords[i][0], drivers.coords[i][1], marker='o', markersize=5, color="green")


    if(save):
        plt.savefig("images/image{0}.png".format(count))
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_paths():
    agents.history = np.array(agents.history).reshape(num_sheep, -1, 2)
    for i in range(num_sheep):
        plt.plot(agents.history[i, :, 0], agents.history[i, :, 1], '--', color="red")
        plt.plot(agents.history[i, 0, 0], agents.history[i, 0, 1], marker='o', markersize=3, color="red")

    d = drivers
    d.history = np.array(d.history).reshape(num_drivers, -1, 2)
    for i in range(num_drivers):
        plt.plot(d.history[i, :, 0], d.history[i, :, 1], ':', color="green")
        plt.plot(d.history[i, 0, 0], d.history[i, 0, 1], marker='o', markersize=3, color="green")

    plt.show()


max_timesteps = 5000
count = 0
reset()
check_radius_initialization()
# agents.timestep()
print(agents.offset())
plot_current()


def terminal():
    p = agents.mean_coords()
    if (p[0] < l and p[1] < l):
        return (True)
    return (False)


while (not terminal()):
    herding_control()
    count += 1
    if (count % 100 == 0):
        print("step",count)
        plot_current()

print(agents.mean_coords(), agents.coords, drivers.coords)
plot_current()
