import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import os
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
from matplotlib import pyplot as plt
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data
import RBM
import rnn_rbm
import time
import midi_manipulation

"""
    This file contains the code for running a tensorflow session to generate music
"""


num = 3 #The number of songs to generate
primer_song = 'Pop_Music_Midi/Every Time We Touch - Chorus.midi' #The path to the song to use to prime the network

def main(saved_weights_path):
    #This function takes as input the path to the weights of the network
    x, cost, generate, W, bh, bv, x, lr, Wuh, Wuv, Wvu, Wuu, bu, u0 = rnn_rbm.rnnrbm()#First we build and get the parameters odf the network

    tvars = [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0]

    saver = tf.train.Saver(tvars) #We use this saver object to restore the weights of the model

    song_primer = midi_manipulation.get_song(primer_song) 

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver.restore(sess, saved_weights_path) #load the saved weights of the network
        # #We generate num songs
        for i in tqdm(range(num)):
            generated_music = sess.run(generate(300), feed_dict={x: song_primer}) #Prime the network with song primer and generate an original song
            new_song_path = "music_outputs/{}_{}".format(i, primer_song.split("/")[-1]) #The new song will be saved here
            midi_manipulation.write_song(new_song_path, generated_music)

if __name__ == "__main__":
    main(sys.argv[1])
    
