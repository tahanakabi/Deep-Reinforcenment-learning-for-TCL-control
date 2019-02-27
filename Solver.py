import pickle
from TCL import TCL
import random
import pandas as pd
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt


# x = np.array([1, 2, 3, 4, 5])
# y = np.array([6, 7, 8, 9, 10])
#
# f = lambda x,y: x*y
# func = np.vectorize(f)
# print(func(x,y))
class MyBounds(object):
    def __init__(self, xmax, xmin ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(np.reshape(x,[24,30]) <= self.xmax))
            tmin = bool(np.all(np.reshape(x,[24,30]) >= self.xmin))
            return tmax and tmin

def compute_cost(TCLs,price):
    return sum([price*TCL.u*TCL.P for TCL in TCLs])

def back_up(u,T):
    # control TCL using u with respect to the backup controller
    if T < Tmin:
        return 1
    elif Tmin < T < Tmax:
        return u
    else:
        return  0

def function(U, T):
    func = np.vectorize(back_up)
    Uphys = func(U,T)
    return Uphys

def g(Uphys,Ti0,Tlist=None, Tmlist=None ):
    # update the indoor and mass temperatures according to (22)
    Tlist1 = []
    Tmlist1 = []
    for k,TCL in enumerate(TCLs):
        T = Tlist[k]
        Tm = Tmlist[k]
        for _ in range(10):
            T = T + TCL.ca * (Ti0 - T) + TCL.cm * (Tm - T) + TCL.P * Uphys[k] + TCL.q
            Tm = Tm + TCL.cm * (T - Tm)
        Tlist1.append(T)
        Tmlist1.append(Tm)
    return Tlist1,Tmlist1

def objective_function(U):
    U=np.reshape(U,[future_steps,num_TCLs])
    T = [TCL.T for TCL in TCLs]
    Tm = [TCL.T for TCL in TCLs]
    Uphyslist =[]
    for i in range(U.shape[0]):
        u = U[i]
        Uphys = function(U=u,T=np.array(T))
        T,Tm = g(Uphys,T0[i],Tlist=T,Tmlist=Tm)
        Uphyslist.append(Uphys)
    Uphys = np.array(Uphyslist)
    cost_per_unit = np.matmul(price,Uphys)
    return np.matmul(p,cost_per_unit)
def binary(u):
    if u<=0.5:
        return 0
    return 1

if __name__=='__main__':
    num_TCLs = 30
    future_steps = 24
    days = 1
    dataframe = pd.read_csv("PriceTemp.csv")
    extTemperatures = dataframe["Temperatures"].values
    prices = dataframe["Prices"].values
    Tmin = 20
    Tmax = 25
    with open('TCLs0.pickle', 'rb') as f:
        TCLs = pickle.load(f)
    start = 0
    for tcl in TCLs:
        tcl.set_T(extTemperatures[start], extTemperatures[start])
    results = []
    actions = []
    bnds = MyBounds(np.zeros(shape=[future_steps,num_TCLs]),np.ones(shape=[future_steps,num_TCLs]))
    x0 = np.zeros(shape=[future_steps, num_TCLs])
    temperatures=[]
    costs = []
    for d in range(days):
        p = np.array([TCL.P for TCL in TCLs])
        index = start + 24*d
        T0 = extTemperatures[index : index+24]
        price = np.array(prices[index : index+24])
        res = basinhopping(func = objective_function, x0=x0, accept_test=bnds)
        us = res.x
        bin = np.vectorize(binary)
        x = bin(us)

        for u in x:
            for i, TCL in enumerate(TCLs):
                TCL.control(u[i])
                TCL.update_state(extTemperatures[index])
            temperatures.append([TCL.T for TCL in TCLs])
            cost = compute_cost(TCLs, prices[index])
            costs.append(cost)
        print(x)
        actions.append(x)

    plt.boxplot(temperatures)
    plt.plot(extTemperatures[:days + 1], label="extTemperatures")
    # plt.plot(prices[:iterations + 1],label="prices")
    plt.plot(costs,label="costs")
    # plt.plot(powers, label="powers")
    plt.legend()

    plt.show()


    with open('solver_results.pickle', 'wb') as f:
        pickle.dump(results, f)
    with open('solver_actions.pickle', 'wb') as f:
        pickle.dump(actions, f)


