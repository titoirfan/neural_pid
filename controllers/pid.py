"""
    This file contains the classic PID controller implementation.
"""

__author__ = "irfantitok@gmail.com"

import numpy as np

class PID(object):
    """
        Classic PID Controller
    """
    def __init__(
        self,
        constants: tuple,
        timestep: float) -> None:
        # Timestep
        self.timestep = timestep

        # Hidden layer connection weights
        self.hidden_weights = np.array([[-1, -1, -1], [1, 1, 1]], dtype=float)
        # Output layer connection weights
        self.output_weights = np.array(constants, dtype=float)

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

    def p_transfer_function(self, v: float) -> float:
        return v

    def i_transfer_function(self, v: float, accumulated_v: float) -> float:
        return v * self.timestep + accumulated_v

    def d_transfer_function(self, v: float, past_v: float) -> float:
        return (v - past_v) / self.timestep

    def predict(self, reference: float, feedback: float) -> float:
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
