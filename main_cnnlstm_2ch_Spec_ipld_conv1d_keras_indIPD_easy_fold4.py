import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import os
import config_2ch_raw_spec_ipld as cfg
from Hat.preprocessing import reshape_3d_to_4d
import prepare_data_2ch_raw_ipd_ild_easy_Spec as pp_data
#from prepare_data import load_data


import keras

from keras.datasets import mnist, cifar10
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.utils import np_utils
from keras.layers import Merge, Input
from keras.regularizers import l1, l2, l1l2, activity_l2
from keras.constraints import nonneg
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
import h5py

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, 6, t_delay, feadim, 1) )

def reshapeX1( X ):
    N = len(X)
    return X.reshape( (N, t_delay, 1, feadim, 1) )

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX2( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim) )

feadim=257
t_delay=33

# hyper-params
fe_fd_left = cfg.dev_fe_mel_fd_left
fe_fd_right = cfg.dev_fe_mel_fd_right
fe_fd_mean = cfg.dev_fe_mel_fd_mean
fe_fd_diff = cfg.dev_fe_mel_fd_diff
fe_fd_ipd = cfg.dev_fe_mel_fd_ipd
fe_fd_ild = cfg.dev_fe_mel_fd_ild

#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 33        # concatenate frames
hop = 10            # step_len
n_hid = 1000
n_out = len( cfg.labels )
print n_out
fold =4         # can be 0, 1, 2, 3, 4

# prepare data
tr_X1, tr_X2, tr_y, te_X1, te_X2, te_y = pp_data.GetAllData_separate( fe_fd_right, fe_fd_left, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, agg_num, hop, fold )
#[batch_num, inmap, n_time, n_freq] = tr_X1.shape
print tr_X1.shape
print tr_X2.shape
#sys.exit()
tr_X1, te_X1 = reshapeX1(tr_X1), reshapeX1(te_X1)
tr_X2, te_X2 = reshapeX1(tr_X2), reshapeX1(te_X2)
    
print tr_X1.shape, tr_X2.shape, tr_y.shape
print te_X1.shape, te_X2.shape, te_y.shape

###build model by keras
kernel_size=(200,1)
pool_size=(257-200+1,1)
#pool_size=(3,1)

model1 = Sequential()
model1.add(TimeDistributed(Convolution2D(128, 200, 1, border_mode='valid', bias=True, W_regularizer=l1l2(l1=0,l2=0)),input_shape=(t_delay,1,feadim, 1)))
model1.add(Activation('relu'))
#model1.add(Dropout(0.1))
model1.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))



model1.add(TimeDistributed(Flatten()))
#model1.add(Reshape((t_delay, feadim),input_shape=(t_delay,feadim)))
print model1.output_shape

#model1.add(Reshape((t_delay, 1, 257, 1), input_shape=(t_delay, 257)))
#print model1.output_shape

#model1.add(TimeDistributed(Convolution2D(128, 200, 1)))
#model1.add(Activation('relu'))
#model1.add(TimeDistributed(MaxPooling2D(pool_size=(257-200+1,1))))
#print model1.output_shape

#model1.add(TimeDistributed(Flatten()))
#print model1.output_shape

model2= Sequential()
model2.add(TimeDistributed(Convolution2D(128, 200, 1, border_mode='valid', bias=True, W_regularizer=l1l2(l1=0,l2=0)),input_shape=(t_delay,1,feadim, 1)))
model2.add(Activation('relu'))
model2.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
model2.add(TimeDistributed(Flatten()))
#model2.add(TimeDistributed(Dense(40),input_shape=(t_delay,40)))
merged = Merge([model1, model2], mode='concat')
model12=Sequential()
model12.add(merged)


model12.add(Bidirectional(GRU(output_dim=128,return_sequences=True,dropout_W=0.0, dropout_U=0.0)))
#print model.output_shape
model12.add(Bidirectional(GRU(output_dim=128,return_sequences=True,dropout_W=0.0, dropout_U=0.0))) # GRU is better than LSTM on refined set, not verified on raw set
model12.add(Bidirectional(GRU(output_dim=128,return_sequences=False,dropout_W=0.0, dropout_U=0.0)))

model12.add(Dense(500))
model12.add(Activation('relu'))
model12.add(Dropout(0.2))

model12.add(Dense(n_out))
model12.add(Activation('sigmoid'))

model12.summary()

#adam=keras.optimizers.Adam(lr=1e-4)
model12.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

dump_fd=cfg.scrap_fd+'/Md/cnn_keras_overlap50_CNN128onSpec_fold4_CNN128onILD257_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')    

model12.fit([tr_X1,tr_X2], tr_y, batch_size=100, nb_epoch=21,
              verbose=1, validation_data=([te_X1,te_X2], te_y), callbacks=[eachmodel]) #, callbacks=[best_model])

#model1.fit(tr_X1, tr_y, batch_size=100, nb_epoch=101,
#              verbose=1, validation_data=(te_X1, te_y), callbacks=[eachmodel]) #, callbacks=[best_model])

#score = model12.evaluate(te_X, te_y, show_accuracy=True, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

