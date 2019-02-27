from NN import NN
from read_data import get_next_state
import tensorflow as tf
import numpy as np
from scipy.optimize import basinhopping
from sklearn.preprocessing import MinMaxScaler
import pickle

class MyBounds(object):
    def __init__(self, xmax=15, xmin=0 ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x < self.xmax))
            tmin = bool(np.all(x > self.xmin))
            return tmax and tmin


def objective_function(x,*args):
    neural_net = args[3]
    xee = np.concatenate((args[1], x.reshape(1, )), axis=0)
    xee = args[4].transform(xee.reshape(1, -1))
    return max(0,neural_net.predict(args[0], xee, args[2]))

if __name__ == '__main__':
    steps_back = 7
    future_steps = 24
    xb, xe, next_xb, next_xe, y = get_next_state(steps_back=steps_back)
    scaler1 = {}
    for i in range(xb.shape[1]):
        scaler1[i] = MinMaxScaler(feature_range=(0,1), copy=True)
        xb[:,i,:] = scaler1[i].fit_transform(xb[:,i,:])
        next_xb[:,i,:] = scaler1[i].fit_transform(next_xb[:,i,:])

    scaler2 = MinMaxScaler(feature_range=(0,1), copy=True).fit(xe)
    xe = scaler2.transform(xe)
    # scaler22 = MinMaxScaler(feature_range=(0,1), copy=True).fit(next_xe)
    scaler3 = MinMaxScaler(feature_range=(0, 1), copy=True).fit(y.reshape(-1,1))


    # prediction = neural_net.prediction

    bnds = MyBounds()
    for n in range(future_steps-1):
        graph1 = tf.Graph()
        neural_net1 = NN(batch_size=1, graph=graph1)
        neural_net1.combined_net(graph=graph1)
        with tf.Session(graph=graph1) as sess:
            saver = tf.train.Saver()
            name = "./model"+str(n)+".ckpt"
            saver.restore(sess, name)
            Q_batch = []
            xb_batch = []
            xe_batch = []
            for l in range(len(y)):
                cl = y[l]
                next_xbl = next_xb[l]
                next_xel = next_xe[l]
                nextQ = basinhopping(func=objective_function, x0=5.0,niter=50,  stepsize=10.0, minimizer_kwargs={"method":"L-BFGS-B", "args":(next_xbl, next_xel, sess, neural_net1,scaler2)}, accept_test=bnds).fun

                Q = cl + scaler3.inverse_transform(np.array(nextQ).reshape(-1,1))[0][0]
                Q_batch.append(Q)
                xb_batch.append(xb[l])
                xe_batch.append(xe[l])
                # print("Xb:")
                # print(xb[l])
                # print("Xe:")
                # print(xe[l])
                print("Q:")
                print(Q)
            Q_batch = np.array(Q_batch)
            scaler3 = MinMaxScaler(feature_range=(0, 1), copy=True).fit(Q_batch.reshape(-1,1))
            Q_batch = scaler3.transform(Q_batch.reshape(-1,1))
            xb_batch = np.array(xb_batch)
            xe_batch = np.array(xe_batch)
        graph2 = tf.Graph()
        neural_net2 = NN(batch_size=990, graph=graph2)
        neural_net2.combined_net(graph=graph2)
        name1 = "./model" + str(n+1) + ".ckpt"
        neural_net2.train(xb_batch,xe_batch,Q_batch, graph=graph2, name=name1)


    with open('scaler.pickle', 'wb') as f:
        pickle.dump(scaler3, f)









