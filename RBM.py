import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pandas as pd


"""
    This file contains the TF implementation of the Restricted Boltzman Machine
"""


#This function lets us easily sample from a vector of probabilities
def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

#This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(x, W, bv, bh, k):
    #Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        #Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) #Propagate the visible values to sample the hidden values
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)) #Propagate the hidden values to sample the visible values
        return count+1, k, xk

    #Run gibbs steps for k iterations
    ct = tf.constant(0) #counter
    [_, _, x_sample] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x], 1, False)
    #We need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

def get_free_energy_cost(x, W, bv, bh, k):   
    #We use this function in training to get the free energy cost of the RBM. We can pass this cost directly into TensorFlow's optimizers 
    #First, draw a sample from the RBM
    x_sample   = gibbs_sample(x, W, bv, bh, k)

    def F(xx):
        #The function computes the free energy of a visible vector. 
        return -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(xx, W) + bh)), 1) - tf.matmul(xx, tf.transpose(bv))

    #The cost is based on the difference in free energy between x and xsample
    cost = tf.reduce_mean(tf.sub(F(x), F(x_sample)))
    return cost

def get_cd_update(x, W, bv, bh, k, lr):
    #This is the contrastive divergence algorithm. 

    #First, we get the samples of x and h from the probability distribution
    #The sample of x
    x_sample = gibbs_sample(x, W, bv, bh, k)
    #The sample of the hidden nodes, starting from the visible state of x
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
    #The sample of the hidden nodes, starting from the visible state of x_sample
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    #Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
    lr = tf.constant(lr, tf.float32) #The CD learning rate
    size_bt = tf.cast(tf.shape(x)[0], tf.float32) #The batch size
    W_  = tf.mul(lr/size_bt, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    bv_ = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(x, x_sample), 0, True))
    bh_ = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(h, h_sample), 0, True))

    #When we do sess.run(updt), TensorFlow will run all 3 update steps
    updt = [W.assign_add(W_), bv.assign_add(bv_), bh.assign_add(bh_)]
    return updt

