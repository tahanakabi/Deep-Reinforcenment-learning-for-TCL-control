import numpy as np
import tensorflow as tf
from read_data import get_X_y
from sklearn.preprocessing import  MinMaxScaler
import matplotlib.pyplot as plt
import pickle


class NN():
    def __init__(self, batch_size = 300, graph = tf.get_default_graph(),test_size = 0.1, steps_back=8, num_TCL=30):
        self.num_TCL = num_TCL
        with graph.as_default():
            # Training Parameters
            self.learning_rate = 0.1
            self.num_steps = 100000
            self.steps_back = steps_back
            self.batch_size = batch_size
            if batch_size==1:
                self.test_proportion = 0
            else:
                self.test_proportion = test_size

            self.batch_tr_size = int(self.batch_size * (1 - self.test_proportion))

            self.test_size = int(self.test_proportion*self.batch_size)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout (keep probability)
            # display_step = 10

            # Network Parameters
            self.cnn_num_input = num_TCL # MNIST data input
            self.fc_num_input = 4
            self.num_output = 1 # MNIST total classes (0-9 digits)
            self.dropout = 0.85 # Dropout, probability to keep units

            # Placeholders
            self.Xb = tf.placeholder(tf.float32, [self.batch_tr_size, self.steps_back, self.cnn_num_input],name='Xb')
            self.Xe = tf.placeholder(tf.float32, [self.batch_tr_size, 1, 4], name='Xe')
            self.Y = tf.placeholder(tf.float32, [self.batch_tr_size, self.num_output], name='Y')

            if self.test_proportion != 0:
                # Test Placeholders
                self.Xb_test = tf.placeholder(tf.float32, [self.test_size, self.steps_back, self.cnn_num_input],name='Xb_test')
                self.Xe_test = tf.placeholder(tf.float32, [self.test_size, 1, 4], name='Xe_test')
                self.Y_test = tf.placeholder(tf.float32, [self.test_size, self.num_output], name='Y_test')

            # Store layers weight & bias
            self.weights = {
                # 5x5 conv
                'wc1': tf.Variable(tf.random_normal([2, 8, 1, 32])),
                # 5x5 conv, 32 inputs, 64 outputs
                'wc2': tf.Variable(tf.random_normal([2, 8, 32, 64])),
                # fully connected for cnn
                'wd1': tf.Variable(tf.random_normal([self.steps_back*self.cnn_num_input*64//4, 1024])),
                'wd11': tf.Variable(tf.random_normal([1024, 20])),
                # fully connected for fl_net,
                'wd2': tf.Variable(tf.random_normal([4, 20])),
                # 1024+10 inputs, 1 output (class prediction)
                'out': tf.Variable(tf.random_normal([20+20, 50])),
                # second fuly connected layer 100 inputs and 1 output
                'out2': tf.Variable(tf.random_normal([50, self.num_output]))
            }

            self.biases = {
                'bc1': tf.Variable(tf.random_normal([32])),
                'bc2': tf.Variable(tf.random_normal([64])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'bd11': tf.Variable(tf.random_normal([20])),
                'bd2': tf.Variable(tf.random_normal([20])),
                'out': tf.Variable(tf.random_normal([50])),
                'out2': tf.Variable(tf.random_normal([self.num_output]))
            }

    # Create some wrappers for simplicity
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    # Create model
    def conv_net(self,xb):
        xb = tf.reshape(xb, shape=[-1, self.steps_back, self.num_TCL, 1])
        # Convolution Layer
        conv1 = self.conv2d(xb, self.weights['wc1'],self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)
        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        # conv2 = self.maxpool2d(conv2, k=2)
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        conv2_reshaped = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(conv2_reshaped, self.weights['wd1']), self.biases['bd1'])
        fc1_relued = tf.nn.relu(fc1)
        fc11 = tf.add(tf.matmul(fc1_relued, self.weights['wd11']), self.biases['bd11'])
        fc11_relued = tf.nn.relu(fc11)
        ## Apply Dropout
        return tf.nn.dropout(fc11_relued, self.keep_prob)


    def fc_net(self,xe):
        xe = tf.reshape(xe, shape=[-1, self.weights['wd2'].get_shape().as_list()[0]])
        fc2 = tf.add(tf.matmul(xe, self.weights['wd2']), self.biases['bd2'])
        return tf.nn.relu(fc2)


    def combined_net(self, graph = tf.get_default_graph()):
        with graph.as_default():
            conv_component = self.conv_net(self.Xb)
            fc_component = self.fc_net(self.Xe)
            # concatenate the to components
            fc = tf.concat([conv_component,fc_component], axis=1)
            # another fc net with sigmoid
            fc3 = tf.add(tf.matmul(fc, self.weights['out']), self.biases['out'])
            fc3_sigmoided = tf.nn.sigmoid(fc3)
            #linear fc
            prediction = tf.add(tf.matmul(fc3_sigmoided, self.weights['out2']), self.biases['out2'], name="prediction")
            # Define loss and optimizer
            loss_op = tf.losses.mean_squared_error(predictions = prediction ,labels = self.Y)
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            train_op = optimizer.minimize(loss_op,name="train_op")

            if self.test_proportion != 0:
                # Test graph
                conv_component_test = self.conv_net(graph.get_tensor_by_name("Xb_test:0"))
                fc_component_test = self.fc_net(graph.get_tensor_by_name("Xe_test:0"))
                # concatenate the to components
                fc_test = tf.concat([conv_component_test, fc_component_test], axis=1)
                # another fc net with sigmoid
                fc3_test = tf.add(tf.matmul(fc_test, self.weights['out']), self.biases['out'])
                fc3_sigmoided_test = tf.nn.sigmoid(fc3_test)
                # linear fc
                prediction_test = tf.add(tf.matmul(fc3_sigmoided_test, self.weights['out2']), self.biases['out2'], name="prediction_test")
                loss_op_test = tf.losses.mean_squared_error(predictions=prediction_test, labels=self.Y_test)


    def run_sess(self, sess, batch_xb, batch_xe, batch_y, saver, name):
        graph = sess.graph

        batch_xe = np.reshape(batch_xe,[-1,1,self.fc_num_input])
        batch_xb = np.reshape(batch_xb, [-1, self.steps_back, self.cnn_num_input])
        batch_y = np.reshape(batch_y,[-1,self.num_output])

        batch_tr_xe = batch_xe[:self.batch_tr_size]
        batch_test_xe = batch_xe[self.batch_tr_size:]

        batch_tr_xb = batch_xb[:self.batch_tr_size]
        batch_test_xb = batch_xb[self.batch_tr_size:]

        batch_tr_y = batch_y[:self.batch_tr_size]
        batch_test_y = batch_y[self.batch_tr_size:]
        overfitting=0
        for step in range(1, self.num_steps + 1):
            # Run optimization op (backprop)
            sess.run("train_op", feed_dict={graph.get_tensor_by_name("Xb:0"): batch_tr_xb,
                                            graph.get_tensor_by_name("Xe:0"): batch_tr_xe,
                                            graph.get_tensor_by_name("Y:0"): batch_tr_y,
                                            graph.get_tensor_by_name("keep_prob:0"): self.dropout})

            # Calculate batch loss

            training_l = sess.run("mean_squared_error/value:0",
                                  feed_dict={graph.get_tensor_by_name("Xb:0"): batch_tr_xb,
                                             graph.get_tensor_by_name("Xe:0"): batch_tr_xe,
                                             graph.get_tensor_by_name("Y:0"): batch_tr_y,
                                             graph.get_tensor_by_name("keep_prob:0"): 1.0})

            test_l = sess.run("mean_squared_error_1/value:0",
                              feed_dict={graph.get_tensor_by_name("Xb_test:0"): batch_test_xb,
                                         graph.get_tensor_by_name("Xe_test:0"): batch_test_xe,
                                         graph.get_tensor_by_name("Y_test:0"): batch_test_y,
                                         graph.get_tensor_by_name("keep_prob:0"): 1.0})


            if step % 10 == 0 or step == 1:
                print("Step " + str(step) +  ", Minibatch training Loss= " + str(training_l))
                print("Step " + str(step) + ", Minibatch validation Loss= " + str(test_l))

            if test_l - training_l> 0.015:
                overfitting += 1
            else: overfitting = 0

            if overfitting >= 30 and training_l <= 0.01 :
                print("condition satisfied")
                break
            if test_l < 0.009 and training_l < 0.009 :
                print("condition satisfied")
                break

                # self.training_loss.append(training_l)
                # self.validation_loss.append(test_l)

        print("Optimization Finished!")
        # Save the variables to disk.
        save_path = saver.save(sess, name)
        print("Model saved in path: %s" % save_path)

    def train(self,xb, xe, y, name = "./model0.ckpt", graph = tf.get_default_graph() ):
        self.training_loss = []
        self.validation_loss = []

        with tf.Session(graph=graph) as sess:
            saver = tf.train.Saver()
            try:
                saver.restore(sess, name)
            except:
                sess.run(tf.global_variables_initializer())
            for i in range(xb.shape[0]//self.batch_size):
                # Run the initializer
                index = i*self.batch_size
                self.run_sess(sess, xb[index:index+self.batch_size],xe[index:index+self.batch_size],y[index:index+self.batch_size], saver, name= name)
        # plt.plot(range(len(self.training_loss)), self.training_loss, label='Training')
        # plt.plot(range(len(self.validation_loss)), self.validation_loss, label='Validation')
        # plt.xlabel('Steps')
        # # plt.ylabel('Loss')
        #
        # plt.title("Loss function")
        #
        # plt.legend()
        #
        # plt.show()

    # def retrain(self,xb, xe, y,sess):
    #     saver.restore(sess, "./model.ckpt")
    #     self.run_sess(sess,xb,xe,y)

    def predict(self, xb, xe, sess):
        # tf Graph input
        graph = sess.graph
        xb = np.reshape(xb, [-1, self.steps_back, self.cnn_num_input])
        xe = np.reshape(xe, [-1, 1, self.fc_num_input])
        p = sess.run("prediction:0", feed_dict={graph.get_tensor_by_name("Xb:0"): xb, graph.get_tensor_by_name("Xe:0"): xe, graph.get_tensor_by_name("keep_prob:0"): 1.0})
        return p

if __name__ == '__main__':
    xb, xe, y = get_X_y(steps_back=7, filename="Q_data0.csv")
    neural_net = NN(batch_size = 100, steps_back=8)
    scaler1 = {}
    for i in range(xb.shape[1]):
        scaler1[i] = MinMaxScaler(feature_range=(0,1), copy=True)
        xb[:,i,:] = scaler1[i].fit_transform(xb[:,i,:])

    scaler2 = MinMaxScaler(feature_range=(0,1), copy=True).fit(xe)
    scaler3 = MinMaxScaler(feature_range=(0, 1), copy=True).fit(y.reshape(-1,1))

    xe=  scaler2.transform(xe)
    y= scaler3.transform(y.reshape(-1,1))
    # graph = tf.Graph()
    neural_net.combined_net()
    # saver = tf.train.Saver()
    # keep_prob = neural_net.keep_prob
    # init = tf.global_variables_initializer()
    # graph = tf.get_default_graph()

    neural_net.train(xb, xe, y)

