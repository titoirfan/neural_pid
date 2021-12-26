"""
    This file contains a usage example of the controllers.
"""

__author__ = "irfantitok@gmail.com"

import numpy as np
import matplotlib.pyplot as plt

from plants.motor_position import MotorPosition
from utils.evaluator import *
from controllers.pidnn import PIDNN
from controllers.pid import PID

# Simulation parameters
t_end = 20
d_t = 0.01
T = np.arange(0.0, t_end, d_t)
it = np.nditer(T, flags=['f_index'])
r = np.full_like(T, 1.0)

num_changes = 2
interval = int(len(T) / num_changes)

for i in range(num_changes):
    r[i * interval:(i + 1) * interval] = i + 1

# Motor position model
plant = MotorPosition(
    K = 1, 
    tau = 0.5)

# Kp, Ki, Kd
constants = [22.34, 7.579, 1.956]

# PIDNN Controller
pidnn = PIDNN(
    initial_constants = constants,
    learning_rate = 1, 
    max_weight_change = 100, 
    tolerance = 1e-8,
    timestep = d_t)

pid = PID(
    constants = constants,
    timestep = d_t)

# Simulate response
y_pidnn = [0.0]
y_pid = [0.0]

# Adaptive
plant.reset_states()
it.reset()
for i in it:
    u = pidnn.predict(reference = r[it.index], feedback = y_pidnn[-1])
    y_pidnn.append(plant.simulate_one_step(u, i))

# Non-adaptive
plant.reset_states()
it.reset()
for i in it:
    u = pid.predict(reference = r[it.index], feedback = y_pid[-1])
    y_pid.append(plant.simulate_one_step(u, i))

# Tweak lengths to match the time series
y_pidnn = y_pidnn[1:]
y_pid = y_pid[1:]

# Performance metrics
settling_time_pidnn = calc_settling_time(y_pidnn, T)
rise_time_pidnn = calc_rise_time(y_pidnn, T)
overshoot_pidnn = calc_overshoot_percent(y_pidnn)

settling_time_pid = calc_settling_time(y_pid, T)
rise_time_pid = calc_rise_time(y_pid, T)
overshoot_pid = calc_overshoot_percent(y_pid)

print("PIDNN - TS: {:.2f} s - TR: {:.2f} s - OV: {:.2f}%".format(
    settling_time_pidnn, 
    rise_time_pidnn, 
    overshoot_pidnn))

print("Classic PID - TS: {:.2f} s - TR: {:.2f} s - OV: {:.2f}%".format(
    settling_time_pid, 
    rise_time_pid, 
    overshoot_pid))

# Plot the results
plt.plot(T, r, 'k--', label = 'Reference')
plt.plot(T, y_pidnn, 'c', label = 'PIDNN')
plt.plot(T, y_pid, 'm', label = 'Classic PID')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend(loc = 'lower right', shadow = False, fontsize = 'medium')
plt.show(block = True)
