Note: this is described in detail here: http://danshiebler.com/2016-08-17-musical-tensorflow-part-two-the-rnn-rbm/


# Music_RNN_RBM

This repository contains code for generating long sequences of polyphonic music by using an RNN_RBM in TensorFlow. 

### TLDR:
You can generate music by cloning the directory and running:
```
python rnn_rbm_generate.py parameter_checkpoints/pretrained.ckpt
```
This will populate the music_outputs directory with midi files that you can play with an application like GuitarBand.

### Training
To train the model, first run the following command to initialize the parameters of the RBM.
```
python weight_initializations.py
```
Then, run the following command to train the RNN_RBM model:
```
python rnn_rbm_train.py <num_epochs>
```
`num_epochs` can be any integer. Set it between 50-500, depending on the hyperparameters.

### Generation:
The command:
```
python rnn_rbm_generate.py <path_to_ckpt_file>
```
will generate music by using the weights stored in the `path_to_ckpt_file`. You can use the provided file `parameter_checkpoints/pretrained.ckpt`, or you can use one of the ckpt files that you create. When you run `train_rnn_rbm.py`, the model creates a `epoch_<x>.ckpt` file in the parameter_checkpoints directory every couple of epochs. 


