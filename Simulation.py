from TCL import TCL
import random
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import pickle
from scipy.optimize import basinhopping
iterations = 1000
num_TCLs = 30
import matplotlib.pyplot as plt

def createTCLs(n, T0):
    # initialize a list of TCLs
    TCLs=[]
    for i in range(n):
        ca = random.normalvariate(0.004,0.0008)
        cm = random.normalvariate(0.2, 0.004)
        q = random.normalvariate(0,0.01)
        P = random.normalvariate(0.5, 0.01)
        tcl= TCL(ca,cm,q,P)
        tcl.set_T(T0,T0)
        TCLs.append(tcl)
    return TCLs

def compute_cost(TCLs,price):
    return sum([price*TCL.u*TCL.P for TCL in TCLs])

def distributeU(U,TCLs):
    pr0 = 2.0
    solution = basinhopping(func=objective_function, x0=pr0,  stepsize=1.0, minimizer_kwargs={"method":"L-BFGS-B", "args":(U)})
    pr = solution.x
    u = [fbid(pr, TCL.SoC) for TCL in TCLs]
    return u

def fbid(pr, SoC):
    pc = 1-SoC
    if pc>pr:
        return 1
    return 0

def objective_function(x,*args):

    return abs(sum([TCL.P for TCL in TCLs if (1-TCL.SoC-x)>0])-args[0])

if __name__ == '__main__':
    dataframe = pd.read_csv("PriceTemp.csv")
    extTemperatures = dataframe["Temperatures"].values
    prices = dataframe["Prices"].values
    # initiate TCLs
    TCLs = createTCLs(num_TCLs, extTemperatures[0])
    # Compute the states of charge
    for TCL in TCLs:
        TCL.compute_SoC()
    # The state vector contains the states of charge of TCLs + outdoor temperature + electricity prices + the hour of the day
    next_state = [TCL.SoC for TCL in TCLs]
    next_state.extend([extTemperatures[0],prices[0], 0])
    # Lists to save and display the results
    mapping = []
    temperatures = []
    costs=[]
    powers = []
    temperatures.append([TCL.Tm for TCL in TCLs])
    for k in range(iterations)[1:]:
        state = next_state
        # choose a random action
        U = 15*random.random()
        # distribute the energy according to TCLs priority
        u = distributeU(U,TCLs)
        # Control each TCL and update their state
        for i,TCL in enumerate(TCLs):
            TCL.control(u[i])
            TCL.update_state(extTemperatures[k-1])
        # save the temperatures of each TCL
        temperatures.append([TCL.T for TCL in TCLs])
        next_state = [TCL.SoC for TCL in TCLs]
        next_state.extend([extTemperatures[k], prices[k], k % 24])
        cost = compute_cost(TCLs, prices[k-1])
        power = sum([TCL.u*TCL.P for TCL in TCLs])
        costs.append(cost)
        powers.append(power)
        # append to the state vector the action U and the cost, then concatenate the next state to it
        state.extend([U,cost])
        state.extend(next_state)
        mapping.append(state)
    temperatures = np.array(temperatures).transpose()
    plt.boxplot(temperatures)
    plt.plot(extTemperatures[:iterations+1], label="extTemperatures")
    # plt.plot(prices[:iterations + 1],label="prices")
    # plt.plot(costs,label="costs")
    plt.plot(powers, label="powers")
    plt.legend()

    plt.show()

    output = np.array(mapping)
    header = range(num_TCLs*2+8)
    df = pd.DataFrame(output,columns=header)
    df.to_csv("Q_data0.csv", index=False,header=True)
    for TCL in TCLs:
        print(TCL.T)
    with open('TCLs0.pickle', 'wb') as f:
        pickle.dump(TCLs, f)



