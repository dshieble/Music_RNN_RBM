import tensorflow as tf
import numpy as np
import glob

from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

import RBM
import midi_manipulation


"""
    This file contains the TF implementation of the RNN-RBM, as well as the hyperparameters of the model
"""

note_range         = midi_manipulation.span #The range of notes that we can produce
n_visible          = 2*note_range*midi_manipulation.num_timesteps #The size of each data vector and the size of the RBM visible layer
n_hidden           = 50 #The size of the RBM hidden layer
n_hidden_recurrent = 100 #The size of each RNN hidden layer

def rnnrbm():

    #This function builds the RNN-RBM and returns the parameters of the model

    x  = tf.placeholder(tf.float32, [None, n_visible]) #The placeholder variable that holds our data
    lr  = tf.placeholder(tf.float32) #The learning rate. We set and change this value during training.
    
    size_bt = tf.shape(x)[0] #the batch size

    #Here we set aside the space for each of the variables.
    #We intialize these variables when we load saved parameters in rnn_rbm_train.py or rnn_rbm_generate.py
    W   = tf.Variable(tf.zeros([n_visible, n_hidden]), name="W")
    Wuh = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden]), name="Wuh")
    Wuv = tf.Variable(tf.zeros([n_hidden_recurrent, n_visible]), name="Wuv")
    Wvu = tf.Variable(tf.zeros([n_visible, n_hidden_recurrent]), name="Wvu")
    Wuu = tf.Variable(tf.zeros([n_hidden_recurrent, n_hidden_recurrent]), name="Wuu")
    bh  = tf.Variable(tf.zeros([1, n_hidden]), name="bh")
    bv  = tf.Variable(tf.zeros([1, n_visible]), name="bv")
    bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="bu")
    u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent]), name="u0")
    BH_t = tf.Variable(tf.zeros([1, n_hidden]), name="BH_t")
    BV_t = tf.Variable(tf.zeros([1, n_visible]), name="BV_t")


    def rnn_recurrence(u_tm1, sl):
        #Iterate through the data in the batch and generate the values of the RNN hidden nodes
        sl  =  tf.reshape(sl, [1, n_visible])
        u_t = (tf.tanh(bu + tf.matmul(sl, Wvu) + tf.matmul(u_tm1, Wuu)))
        return u_t

    def visible_bias_recurrence(bv_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the visible bias vectors
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        return bv_t

    def hidden_bias_recurrence(bh_t, u_tm1):
        #Iterate through the values of the RNN hidden nodes and generate the values of the hidden bias vectors
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))
        return bh_t       

    def generate_recurrence(count, k, u_tm1, primer, x, music):
        #This function builds and runs the gibbs steps for each RBM in the chain to generate music
        #Get the bias vectors from the current state of the RNN
        bv_t = tf.add(bv, tf.matmul(u_tm1, Wuv))
        bh_t = tf.add(bh, tf.matmul(u_tm1, Wuh))

        #Run the Gibbs step to get the music output. Prime the RBM with the previous musical output.
        x_out = RBM.gibbs_sample(primer, W, bv_t, bh_t, k=25)
        
        #Update the RNN hidden state based on the musical output and current hidden state.
        u_t  = (tf.tanh(bu + tf.matmul(x_out, Wvu) + tf.matmul(u_tm1, Wuu)))

        #Add the new output to the musical piece
        music = tf.concat(0, [music, x_out])
        return count+1, k, u_t, x_out, x, music

    def generate(num, x=x, size_bt=size_bt, u0=u0, n_visible=n_visible, prime_length=100):
        """
            This function handles generating music. This function is one of the outputs of the build_rnnrbm function
            Args:
                num (int): The number of timesteps to generate
                x (tf.placeholder): The data vector. We can use feed_dict to set this to the music primer. 
                size_bt (tf.float32): The batch size
                u0 (tf.Variable): The initial state of the RNN
                n_visible (int): The size of the data vectors
                prime_length (int): The number of timesteps into the primer song that we use befoe beginning to generate music
            Returns:
                The generated music, as a tf.Tensor

        """
        Uarr = tf.scan(rnn_recurrence, x, initializer=u0)
        U = Uarr[np.floor(prime_length/midi_manipulation.num_timesteps), :, :]
        [_, _, _, _, _, music] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                                         generate_recurrence, [tf.constant(1, tf.int32), tf.constant(num), U,
                                                         tf.zeros([1, n_visible], tf.float32), x, 
                                                         tf.zeros([1, n_visible],  tf.float32)])
        return music

    #Reshape our bias matrices to be the same size as the batch.
    tf.assign(BH_t, tf.tile(BH_t, [size_bt, 1]))
    tf.assign(BV_t, tf.tile(BV_t, [size_bt, 1]))
    #Scan through the rnn and generate the value for each hidden node in the batch
    u_t  = tf.scan(rnn_recurrence, x, initializer=u0)
    #Scan through the rnn and generate the visible and hidden biases for each RBM in the batch
    BV_t = tf.reshape(tf.scan(visible_bias_recurrence, u_t, tf.zeros([1, n_visible], tf.float32)), [size_bt, n_visible])
    BH_t = tf.reshape(tf.scan(hidden_bias_recurrence, u_t, tf.zeros([1, n_hidden], tf.float32)), [size_bt, n_hidden])
    #Get the free energy cost from each of the RBMs in the batch 
    cost = RBM.get_free_energy_cost(x, W, BV_t, BH_t, k=15)
    return x, cost, generate, W, bh, bv, x, lr, Wuh, Wuv, Wvu, Wuu, bu, u0

