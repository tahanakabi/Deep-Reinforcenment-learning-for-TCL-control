import pandas as pd
import numpy as np

def read_csv_to_df(path):
    datarame = pd.read_csv(path)
    datarame.dropna()
    return datarame

def get_X_y(steps_back, filename, num_TCL=30):
    data = read_csv_to_df(filename)
    data = data.ix[:,:num_TCL+5]
    xb = []
    xe = []
    y = []
    for i in range(data.shape[0])[:-steps_back]:
        xb.append(np.array(data.ix[i:i+steps_back,:num_TCL].values))
        xe.append(np.array(data.ix[i+steps_back,num_TCL:num_TCL+4].values))
        # print(xe)
        y.append(np.array(data.ix[i+steps_back,-1]))
    xb = np.array(xb)
    xe = np.array(xe)
    y = np.array(y)
    return xb,xe, y

def get_next_state(steps_back,num_TCL=30):
    data = read_csv_to_df("Q_data0.csv")
    xb = []
    next_xb = []
    xe = []
    next_xe = []
    y = []
    for i in range(data.shape[0])[:-steps_back]:
        xb.append(np.array(data.ix[i:i+steps_back,:num_TCL].values))
        xe.append(np.array(data.ix[i+steps_back,num_TCL:num_TCL+4].values))
        # print(xe)
        y.append(np.array(data.ix[i+steps_back,num_TCL+4]))

        next_xb.append(np.array(data.ix[i:i + steps_back, num_TCL+5:num_TCL*2+5].values))
        next_xe.append(np.array(data.ix[i + steps_back, num_TCL*2+5:].values))

    xb = np.array(xb)
    xe = np.array(xe)
    next_xb = np.array(next_xb)
    next_xe = np.array(next_xe)
    y = np.array(y)
    return xb,xe, next_xb, next_xe, y




