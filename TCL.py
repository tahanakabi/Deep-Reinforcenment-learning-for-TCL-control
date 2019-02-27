from random import random

class TCL:
    def __init__(self, ca, cm, q, P, Tmin=20, Tmax=25):
        self.ca = ca
        self.cm = cm
        self.q = q
        self.P = P
        self.Tmin = Tmin
        self.Tmax = Tmax


    def set_T(self, T, Tm):
        self.T = T
        self.Tm = Tm

    def control(self, ui=0):
        # control TCL using u with respect to the backup controller
        if self.T < self.Tmin:
            self.u = 1
        elif self.Tmin<self.T<self.Tmax:
            self.u = ui
        else:
            self.u = 0

    def update_state(self, T0):
        # update the indoor and mass temperatures according to (22)
        for _ in range(10):
            # if self.T > self.Tmax:
            #     self.u = 0
            self.T = self.T + self.ca * (T0 - self.T) + self.cm * (self.Tm - self.T) + self.P * self.u +self.q
            self.Tm = self.Tm + self.cm*(self.T - self.Tm)
        self.compute_SoC()

    def compute_SoC(self):
        self.SoC = (self.T-self.Tmin)/(self.Tmax-self.Tmin)