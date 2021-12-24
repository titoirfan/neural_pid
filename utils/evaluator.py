import numpy as np

def calc_settling_time(
    y: list(), 
    t: list(),
    settling_time_threshold = 0.02) -> float:
    steady_state_val = y[-1]
    upper_margin = (1.0 + settling_time_threshold) * steady_state_val
    lower_margin = (1.0 - settling_time_threshold) * steady_state_val

    for i in reversed(range(len(t))):
        if y[i] <= lower_margin or y[i] >= upper_margin:
            settling_time = t[i]
            break

    return settling_time 

def calc_rise_time(
    y: list(),
    t: list(),
    rise_time_lower_val_bound = 0.1,
    rise_time_upper_val_bound = 0.9) -> float:
    steady_state_val = y[-1]

    rise_time_lower_idx = (np.where(y >= rise_time_lower_val_bound * steady_state_val)[0])[0]
    rise_time_upper_idx = (np.where(y >= rise_time_upper_val_bound * steady_state_val)[0])[0]

    rise_time = t[rise_time_upper_idx] - t[rise_time_lower_idx]

    return rise_time

def calc_overshoot_percent(y: list()) -> float:
    steady_state_val = y[-1]
    overshoot_val = max(y) / steady_state_val - 1
    overshoot_percent = overshoot_val * 100

    return overshoot_percent