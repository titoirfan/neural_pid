import numpy as np
import control.matlab

class MotorPosition:
    def __init__(self, K: float, tau: float):
        # System characteristics
        self._K = K
        self._tau = tau
        self._G = control.matlab.TransferFunction([K], [tau, 1, 0])

        # System state
        self._X = 0.0
        self._U = np.array([0.0])
        self._T = np.array([0.0])

    def reset_states(self):
        self._X = 0.0
        self._U = np.array([0.0])
        self._T = np.array([0.0])

    def simulate_one_step(self, U_input, T_input):
        self._T = np.array([self._T[-1], T_input])
        self._U = np.array([self._U[-1], U_input])

        y, _, x = control.matlab.lsim(self._G, self._U, self._T, X0 = self._X)
        self._X = x[-1]

        return y[-1]