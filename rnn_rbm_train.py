import time
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import rnn_rbm
import midi_manipulation 

"""
    This file contains the code for training the RNN-RBM by using the data in the Pop_Music_Midi directory
"""


batch_size = 100 #The number of trianing examples to feed into the rnn_rbm at a time
epochs_to_save = 5 #The number of epochs to run between saving each checkpoint
saved_weights_path = "parameter_checkpoints/initialized.ckpt" #The path to the initialized weights checkpoint file

def main(num_epochs):
    #First, we build the model and get pointers to the model parameters
    x, cost, generate, W, bh, bv, x, lr, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()

    #The trainable variables include the weights and biases of the RNN and the RBM, as well as the initial state of the RNN
    tvars = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]
    # opt_func = tf.train.AdamOptimizer(learning_rate=lr) 
    # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 1)
    # updt = opt_func.apply_gradients(zip(grads, tvars)) 
    
    #The learning rate of the  optimizer is a parameter that we set on a schedule during training
    opt_func = tf.train.GradientDescentOptimizer(learning_rate=lr)
    gvs = opt_func.compute_gradients(cost, tvars)
    gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs] #We use gradient clipping to prevent gradients from blowing up during training
    updt = opt_func.apply_gradients(gvs)#The update step involves applying the clipped gradients to the model parameters

    songs = midi_manipulation.get_songs('Pop_Music_Midi') #Load the songs 

    saver = tf.train.Saver(tvars) #We use this saver object to restore the weights of the model and save the weights every few epochs
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init) 
        saver.restore(sess, saved_weights_path) #Here we load the initial weights of the model that we created with weight_initializations.py

        #We run through all of the songs n_epoch times
        print "starting"
        for epoch in range(num_epochs):
            costs = []
            start = time.time()
            for s_ind, song in enumerate(songs):
                for i in range(1, len(song), batch_size):
                    tr_x = song[i:i + batch_size] 
                    alpha = min(0.01, 0.1/float(i)) #We decrease the learning rate according to a schedule.
                    _, C = sess.run([updt, cost], feed_dict={x: tr_x, lr: alpha}) 
                    costs.append(C) 
            #Print the progress at epoch
            print "epoch: {} cost: {} time: {}".format(epoch, np.mean(costs), time.time()-start)
            print
            #Here we save the weights of the model every few epochs
            if (epoch + 1) % epochs_to_save == 0: 
                saver.save(sess, "parameter_checkpoints/epoch_{}.ckpt".format(epoch))

if __name__ == "__main__":
    main(int(sys.argv[1]))


