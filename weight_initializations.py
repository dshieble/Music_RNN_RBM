import tensorflow as tf
import numpy as np
from tqdm import tqdm
import RBM
import rnn_rbm #The hyperparameters of the RBM and RNN-RBM are specified in the rnn_rbm file
import midi_manipulation 

"""
	This file stores the code for initializing the weights of the RNN-RBM. We initialize the parameters of the RBMs by 
	training them directly on the data with CD-k. We initialize the parameters of the RNN with small weights.
"""

num_epochs = 100 #The number of epochs to train the RBM
lr = 0.01 #The learning rate for the RBM

def main():
	#Load the Songs
	songs = midi_manipulation.get_songs('Pop_Music_Midi')


	x  = tf.placeholder(tf.float32, [None, rnn_rbm.n_visible], name="x") #The placeholder variable that holds our data
	W   = tf.Variable(tf.random_normal([rnn_rbm.n_visible, rnn_rbm.n_hidden], 0.01), name="W") #The weight matrix of the RBM
	Wuh = tf.Variable(tf.random_normal([rnn_rbm.n_hidden_recurrent, rnn_rbm.n_hidden], 0.0001), name="Wuh")  #The RNN -> RBM hidden weight matrix
	bh  = tf.Variable(tf.zeros([1, rnn_rbm.n_hidden], tf.float32), name="bh") #The RNN -> RBM hidden bias vector
	Wuv = tf.Variable(tf.random_normal([rnn_rbm.n_hidden_recurrent, rnn_rbm.n_visible], 0.0001), name="Wuv") #The RNN -> RBM visible weight matrix
	bv  = tf.Variable(tf.zeros([1, rnn_rbm.n_visible], tf.float32), name="bv")#The RNN -> RBM visible bias vector
	Wvu = tf.Variable(tf.random_normal([rnn_rbm.n_visible, rnn_rbm.n_hidden_recurrent], 0.0001), name="Wvu") #The data -> RNN weight matrix
	Wuu = tf.Variable(tf.random_normal([rnn_rbm.n_hidden_recurrent, rnn_rbm.n_hidden_recurrent], 0.0001), name="Wuu") #The RNN hidden unit weight matrix
	bu  = tf.Variable(tf.zeros([1, rnn_rbm.n_hidden_recurrent],  tf.float32), name="bu")   #The RNN hidden unit bias vector
	u0  = tf.Variable(tf.zeros([1, rnn_rbm.n_hidden_recurrent], tf.float32), name="u0") #The initial state of the RNN

	#The RBM bias vectors. These matrices will get populated during rnn-rbm training and generation
	BH_t = tf.Variable(tf.ones([1, rnn_rbm.n_hidden],  tf.float32), name="BH_t") 
	BV_t = tf.Variable(tf.ones([1, rnn_rbm.n_visible],  tf.float32), name="BV_t") 

	#Build the RBM optimization
	saver = tf.train.Saver()

	#Note that we initialize the RNN->RBM bias vectors with the bias vectors of the trained RBM. These vectors will form the templates for the bv_t and
	#bh_t of each RBM that we create when we run the RNN-RBM
	updt = RBM.get_cd_update(x, W, bv, bh, 1, lr)

	#Run the session
	with tf.Session() as sess:
		#Initialize the variables of the model
	    init = tf.initialize_all_variables()
	    sess.run(init)

	    #Run over each song num_epoch times
	    for epoch in tqdm(range(num_epochs)):
	        for song in songs:
	            sess.run(updt, feed_dict={x: song})
	    #Save the initialized model here
	    save_path = saver.save(sess, "parameter_checkpoints/initialized.ckpt")

if __name__ == "__main__":
    main()
