from TCL import TCL
import random
import pandas as pd
from scipy.optimize import minimize
import numpy as np
import pickle
from read_data import get_X_y
from Q_iteration import objective_function as objQ, MyBounds
from scipy.optimize import basinhopping
import tensorflow as tf
from NN import NN
from sklearn.preprocessing import MinMaxScaler
import csv
import os
import matplotlib.pyplot as plt
# def createTCLs(n, T0):
#     # initialize a list of TCLs
#     TCLs=[]
#     for i in range(n):
#         ca = random.normalvariate(0.004,0.0008)
#         cm = random.normalvariate(0.2, 0.004)
#         q = random.normalvariate(0,0.01)
#         P = random.normalvariate(0.5, 0.1)
#         tcl= TCL(ca,cm,q,P)
#         tcl.set_T(T0,T0)
#         TCLs.append(tcl)
#     return TCLs

def compute_cost(TCLs,price):
    return sum([price*TCL.u for TCL in TCLs])

def distributeU(U,TCLs):

    pr0=0.5
    bnds=[(0.0, 1.0)]
    solution = minimize(fun = objective_function, x0 = pr0, args = (U), bounds= bnds)
    pr = solution.x
    u = [fbid(pr, TCL.SoC) for TCL in TCLs]
    return u

def fbid(pr, SoC):
    pc = 1-SoC
    if pc>pr:
        return 1
    return 0

def sum_fbids(pr):
    fbids = [fbid(pr, TCL.SoC)*TCL.P for TCL in TCLs]
    return sum(fbids)

def objective_function(x,*args):
    return abs(sum_fbids(x)-args[0])

def scale(Xb_original,Xb,Xe,Y=None):
    scaler1 = {}
    for i in range(Xb.shape[1]):
        scaler1[i] = MinMaxScaler(feature_range=(0,1), copy=True)
        scaler1[i].fit(Xb_original[:,i,:])
        Xb[:,i,:] = scaler1[i].transform(Xb[:,i,:])

    scaler2 = MinMaxScaler(feature_range=(0,1), copy=True).fit(Xe)

    return scaler1, scaler2



if __name__ == '__main__':

    dataframe = pd.read_csv("PriceTemp.csv")
    extTemperatures = dataframe["Temperatures"].values
    prices = dataframe["Prices"].values
    iterations = 2
    start = 0
    with open('TCLs0.pickle', 'rb') as f:
        TCLs = pickle.load(f)

    with open('scaler.pickle', 'rb') as s:
        scaler = pickle.load(s)

    for TCL in TCLs:
        TCL.set_T(extTemperatures[start], extTemperatures[start])
        TCL.compute_SoC()

    # next_state = [TCL.SoC for TCL in TCLs]
    # next_state.extend([extTemperatures[0],prices[0], 0])
    mapping = []
    with open("Q_data01.csv", "a", newline='') as fp:
         wr = csv.writer(fp, dialect='excel')
         wr.writerow(range(len(TCLs)*2+8))
    costs = []
    temperatures = []

    for k in range(iterations):
        graph1 = tf.Graph()
        neural_net1 = NN(batch_size=1, graph=graph1)
        neural_net1.combined_net(graph=graph1)
        Xb_original, Xe_original, _ = get_X_y(steps_back=7, filename="Q_data0.csv")
        with tf.Session(graph=graph1) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./model23.ckpt")
            for h in range(24):
                step = start + k * 23 + h
                time = step%24
                state = [TCL.SoC for TCL in TCLs]
                state.extend([extTemperatures[step], prices[step], time%24])
                greed = random.random()
                epsilon = 0.2 * (1 / (k+1) ** 0.7)
                if greed < epsilon  or step <= 8:
                # if False:
                    U = 15*random.random()
                else:
                    Xb, Xe, _ = get_X_y(steps_back=7, filename="Q_data01.csv")
                    scaler1, scaler2 = scale(Xb_original, Xb, Xe_original)
                    Xe = Xe[:, :-1]
                    bnds = MyBounds()
                    U = basinhopping(func=objQ, x0=5.0,  stepsize=20.0, minimizer_kwargs={"method":"L-BFGS-B", "args":(Xb[-1], Xe[-1], sess, neural_net1, scaler2)}, accept_test=bnds).x
                    # U = scaler.inverse_transform(np.array(U).reshape(1, -1))
                    U = float(U[0])
                    print(U)
                u = distributeU(U,TCLs)
                for i,TCL in enumerate(TCLs):
                    TCL.control(u[i])
                    TCL.update_state(extTemperatures[step+1])
                temperatures.append([TCL.T for TCL in TCLs])
                next_state = [TCL.SoC for TCL in TCLs]
                next_state.extend([extTemperatures[step+1], prices[step+1], (time+1)%24])
                cost = compute_cost(TCLs, prices[k])
                costs.append(cost)
                print("cost:"+str(cost))
                state.extend([U,cost])
                state.extend(next_state)
                with open("Q_data01.csv", "a", newline='') as fp:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(state)
        temperatures = np.array(temperatures).transpose()
        plt.boxplot(temperatures)
        plt.plot(extTemperatures[:iterations + 1], label="extTemperatures")
        # plt.plot(prices[:iterations + 1],label="prices")
        plt.plot(costs,label="costs")
        # plt.plot(powers, label="powers")
        plt.legend()
        plt.show()

        # os.system('python Q_iteration.py')

