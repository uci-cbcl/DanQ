import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from seya.layers.recurrent import Bidirectional
from keras.utils.layer_utils import print_layer_shapes


print 'loading data'
trainmat = h5py.File('data/train.mat')
validmat = scipy.io.loadmat('data/valid.mat')
testmat = scipy.io.loadmat('data/test.mat')

X_train = np.transpose(np.array(trainmat['trainxdata']),axes=(2,0,1))
y_train = np.array(trainmat['traindata']).T

forward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True)
backward_lstm = LSTM(input_dim=1024, output_dim=512, return_sequences=True)
brnn = Bidirectional(forward=forward_lstm, backward=backward_lstm, return_sequences=True)

print 'building model'

model = Sequential()
conv_layer = Convolution1D(input_dim=4,
                        input_length=1000,
                        nb_filter=1024,
                        filter_length=30,
                        border_mode="valid",
                        activation="relu",
                        subsample_length=1)


conv_weights = conv_layer.get_weights()

JASPAR_motifs = list(np.load('JASPAR_CORE_2016_vertebrates.npy'))

reverse_motifs = [JASPAR_motifs[19][::-1,::-1], JASPAR_motifs[97][::-1,::-1], JASPAR_motifs[98][::-1,::-1], JASPAR_motifs[99][::-1,::-1], JASPAR_motifs[100][::-1,::-1], JASPAR_motifs[101][::-1,::-1]]
JASPAR_motifs = JASPAR_motifs + reverse_motifs

for i in xrange(len(JASPAR_motifs)):
    m = JASPAR_motifs[i][::-1,:]
    w = len(m)
    #conv_weights[0][i,:,:,0] = 0
    #start = (30-w)/2
    start = np.random.randint(low=3, high=30-w+1-3)
    conv_weights[0][i,:,start:start+w,0] = m.T - 0.25
    #conv_weights[1][i] = -0.5
    conv_weights[1][i] = np.random.uniform(low=-1.0,high=0.0)

conv_layer.set_weights(conv_weights)
model.add(conv_layer)

model.add(MaxPooling1D(pool_length=15, stride=15))

model.add(Dropout(0.2))

model.add(brnn)

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(input_dim=64*1024, output_dim=925, activation="relu"))

model.add(Dense(input_dim=925, output_dim=919, activation="sigmoid"))

print 'compiling model'
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary")

print 'running at most 32 epochs'

checkpointer = ModelCheckpoint(filepath="DanQ-JASPAR_bestmodel.hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

model.fit(X_train, y_train, batch_size=100, nb_epoch=32, shuffle=True, show_accuracy=True, validation_data=(np.transpose(validmat['validxdata'],axes=(0,2,1)), validmat['validdata']), callbacks=[checkpointer,earlystopper])

tresults = model.evaluate(np.transpose(testmat['testxdata'],axes=(0,2,1)), testmat['testdata'],show_accuracy=True)

print tresults

