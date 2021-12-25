"""
    This file contains the adaptive PID neural network
    controller implementation.
"""

__author__ = "irfantitok@gmail.com"

import math
import numpy as np
from typing import Any

class PIDNN(object):
    """
        Adaptive PID Neural Network Controller
    """
    def __init__(
        self,
        initial_constants: tuple[float, float, float],
        learning_rate: float,
        max_weight_change: float,
        tolerance: float,
        timestep: float) -> None:
        # The network learning rate
        self.eta = learning_rate
        # Error tolerance
        self.tol = tolerance
        # Maximum weight change
        self.max_weight_chg = max_weight_change
        # Maximum tolerance for weight change
        self.weight_chg_tol = self.max_weight_chg * math.log(2) / self.eta
        # Division by zero tolerance
        self.div_by_zero_tol = 1e-20
        # Timestep
        self.timestep = timestep

        # Hidden layer connection weights
        self.hidden_weights = np.array([[-1, -1, -1], [1, 1, 1]], dtype=float)
        # Output layer connection weights
        self.output_weights = np.array(initial_constants, dtype=float)

        # Indexed according to [t, t - 1]
        # System feedback vector
        self.y = np.zeros([2], dtype=float)
        # Reference vector
        self.r = np.zeros([2], dtype=float)

        # Feedback input vector
        self.input_y = np.zeros([2], dtype=float)
        # Reference input vector
        self.input_r = np.zeros([2], dtype=float)

        # Hidden layer inputs
        self.hidden_input_p = np.zeros([2], dtype=float)
        self.hidden_input_i = np.zeros([2], dtype=float)
        self.hidden_input_d = np.zeros([2], dtype=float)

        # Hidden layer outputs
        self.hidden_output_p = np.zeros([2], dtype=float)
        self.hidden_output_i = np.zeros([2], dtype=float)
        self.hidden_output_d = np.zeros([2], dtype=float)

        # Control effort vector
        self.v = np.zeros([2], dtype=float)

    def threshold_div_by_zero(self, value: float) -> float:
        if math.fabs(value) < self.div_by_zero_tol:
            if value >= 0:
                return self.div_by_zero_tol
            else:
                return -self.div_by_zero_tol
        return value

    def p_transfer_function(self, v: float) -> float:
        return v

    def i_transfer_function(self, v: float, accumulated_v: float) -> float:
        return v * self.timestep + accumulated_v

    def d_transfer_function(self, v: float, past_v: float) -> float:
        return (v - past_v) / self.timestep

    def predict(self, reference: float, feedback: float) -> float:
        # Update weights
        self.learn(feedback)
        
        # Update variable history
        self.y[1] = self.y[0]
        self.r[1] = self.r[0]
        self.y[0] = feedback
        self.r[0] = reference

        self.input_y[1] = self.input_y[0]
        self.input_r[1] = self.input_r[0]
        self.hidden_input_p[1] = self.hidden_input_p[0]
        self.hidden_input_i[1] = self.hidden_input_i[0]
        self.hidden_input_d[1] = self.hidden_input_d[0]
        self.hidden_output_p[1] = self.hidden_output_p[0]
        self.hidden_output_i[1] = self.hidden_output_i[0]
        self.hidden_output_d[1] = self.hidden_output_d[0]

        self.v[1] = self.v[0]

        # Calculate input neuron outputs
        self.input_y[0] = self.p_transfer_function(feedback)
        self.input_r[0] = self.p_transfer_function(reference)

        # Calculate hidden P neurons outputs
        self.hidden_input_p[0] = (
            self.input_y[0] * self.hidden_weights[0][0]
            + self.input_r[0] * self.hidden_weights[1][0])
        self.hidden_output_p[0] = self.p_transfer_function(self.hidden_input_p[0])

        # Calculate hidden I neurons outputs
        self.hidden_input_i[0] = (
            self.input_y[0] * self.hidden_weights[0][1]
            + self.input_r[0] * self.hidden_weights[1][1])
        self.hidden_output_i[0] = self.i_transfer_function(
            self.hidden_input_i[0], self.hidden_output_i[1])

        # Calculate hidden D neurons outputs
        self.hidden_input_d[0] = (
            self.input_y[0] * self.hidden_weights[0][2]
            + self.input_r[0] * self.hidden_weights[1][2])
        self.hidden_output_d[0] = self.d_transfer_function(
            self.hidden_input_d[0], self.hidden_input_d[1])

        # Calculate output neuron outputs
        self.v[0] = (
            self.hidden_output_p[0] * self.output_weights[0]
            + self.hidden_output_i[0] * self.output_weights[1]
            + self.hidden_output_d[0] * self.output_weights[2])

        return self.v[0]

    def learn(self, feedback: float) -> None:
        # Backprop
        delta_r = self.r[0] - self.y[0]
        delta_y = feedback - self.y[0]

        delta_output_weights = self.backprop(delta_r, delta_y)

        # Update weights when error is larger than the tolerance
        if delta_r >= self.tol:
            for idx in range(delta_output_weights.shape[0]):
                if math.fabs(delta_output_weights[idx]) > self.weight_chg_tol:
                    if delta_output_weights[idx] > 0:
                        delta_output_weights[idx] = self.weight_chg_tol
                    elif delta_output_weights[idx] < 0:
                        delta_output_weights[idx] = -self.weight_chg_tol
                
                self.output_weights[idx] = self.output_weights[idx] - self.eta * delta_output_weights[idx]

    def backprop(self, delta_r: float, delta_y: float) -> Any:
        delta_v = self.threshold_div_by_zero(self.v[0] - self.v[1])

        # Output layer weight changes
        delta_output_weights = np.zeros([3], dtype=float)
        delta_output_weights[0] = (
            -2 * delta_r * delta_y * self.hidden_output_p[0] / delta_v)
        delta_output_weights[1] = (
            -2 * delta_r * delta_y * self.hidden_output_i[0] / delta_v)
        delta_output_weights[2] = (
            -2 * delta_r * delta_y * self.hidden_output_d[0] / delta_v)

        return delta_output_weights
