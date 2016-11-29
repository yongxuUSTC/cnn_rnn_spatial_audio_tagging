 
import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats

import keras
from keras.models import load_model

from keras import backend as K

import config_2ch_raw_wav32ms_ipld_eva as cfg
import prepare_data_2ch_raw_ipd_ild_easy as pp_data
import csv
from Hat.preprocessing import reshape_3d_to_4d
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d, reshape_3d_to_4d, mat_concate_multiinmaps6in
from Hat.metrics import prec_recall_fvalue
import cPickle
import eer
import matplotlib.pyplot as plt
#from main_cnn import fe_fd, agg_num, hop, n_hid, fold
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)


# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, 6, t_delay, feadim, 1) )

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX2( X ):
    N = len(X)
    return X.reshape( (N, t_delay, feadim) )

def reshapeX1( X ):
    N = len(X)
    return X.reshape( (N, t_delay, 1, feadim, 1) )

def reshapeX3( X ):
    N = len(X)
    return X.reshape( (N, t_delay, 1, 257, 1) )

feadim=512
t_delay=33


debug=0
# hyper-params
n_labels = len( cfg.labels )
fe_fd_left = cfg.dev_fe_mel_fd_left
fe_fd_right = cfg.dev_fe_mel_fd_right
fe_fd_mean = cfg.dev_fe_mel_fd_mean
fe_fd_diff = cfg.dev_fe_mel_fd_diff
fe_fd_ipd = cfg.dev_fe_mel_fd_ipd
fe_fd_ild = cfg.dev_fe_mel_fd_ild
#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 33        # concatenate frames
hop = 1          # step_len
n_hid = 1000
fold = 9        # can be 0, 1, 2, 3

# load model
# load model
#md = serializations.load( cfg.scrap_fd + '/Md/md20.p' )
md=load_model('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_2ch_wav_ipd_ild_overlap/Md/cnn_keras_overlap50_cnn128onRAW_eva816_CNN128onILD257_weights.18-0.62.hdf5')
#md.summary()

def recognize():
    ## prepare data
    #_, _, te_X, te_y = pp_data.GetAllData(fe_fd_right, fe_fd_left, fe_fd_mean, fe_fd_diff, agg_num, hop, fold )
    ##te_X = reshapeX(te_X)
    #print te_X.shape
    
    # do recognize and evaluation
    thres = 0.4     # thres, tune to prec=recall, if smaller, make prec smaller
    n_labels = len( cfg.labels )
    
    gt_roll = []
    pred_roll = []
    result_roll = []
    y_true_binary_c = []
    y_true_file_c = []
    y_true_binary_m = []
    y_true_file_m = []
    y_true_binary_f = []
    y_true_file_f = []
    y_true_binary_v = []
    y_true_file_v = []
    y_true_binary_p = []
    y_true_file_p = []
    y_true_binary_b = []
    y_true_file_b = []
    y_true_binary_o = []
    y_true_file_o = []
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            if fold==curr_fold:
                # get features, tags
                fe_path_left = fe_fd_left + '/' + na + '.f'
                fe_path_right = fe_fd_right + '/' + na + '.f'
                fe_path_mean = fe_fd_mean + '/' + na + '.f'
                fe_path_diff = fe_fd_diff + '/' + na + '.f'
                fe_path_ipd = fe_fd_ipd + '/' + na + '.f'
                fe_path_ild = fe_fd_ild + '/' + na + '.f'
                #fe_path_ori = fe_fd_ori + '/' + na + '.f'
                info_path = cfg.dev_wav_fd + '/' + na + '.csv'
                #print na
                tags = pp_data.GetTags( info_path )
                #print tags
                y = pp_data.TagsToCategory( tags )
                #print y
                #sys.exit()
                #X_l = cPickle.load( open( fe_path_left, 'rb' ) )
                #X_r = cPickle.load( open( fe_path_right, 'rb' ) )
                X_m = cPickle.load( open( fe_path_mean, 'rb' ) )
                #X_d = cPickle.load( open( fe_path_diff, 'rb' ) )
                #X_ipd = cPickle.load( open( fe_path_ipd, 'rb' ) )
                X_ild = cPickle.load( open( fe_path_ild, 'rb' ) )
                #X_o = cPickle.load( open( fe_path_ori, 'rb' ) )

                # aggregate data
                #X3d_l = mat_2d_to_3d( X_l, agg_num, hop )
                #X3d_r = mat_2d_to_3d( X_r, agg_num, hop )
                X3d_m = mat_2d_to_3d( X_m, agg_num, hop )
   		#X3d_d = mat_2d_to_3d( X_d, agg_num, hop )
                #X3d_ipd = mat_2d_to_3d( X_ipd, agg_num, hop )
   		X3d_ild = mat_2d_to_3d( X_ild, agg_num, hop )
   		#X3d_o = mat_2d_to_3d( X_o, agg_num, hop )
     	        ## reshape 3d to 4d
       	        #X4d_l = reshape_3d_to_4d( X3d_l)
                #X4d_r = reshape_3d_to_4d( X3d_r)
                #X4d_m = reshape_3d_to_4d( X3d_m)
                #X4d_d = reshape_3d_to_4d( X3d_d)
                # concatenate
                #X4d=mat_concate_multiinmaps6in(X3d_l, X3d_r, X3d_m, X3d_d, X3d_ipd, X3d_ild)
                #print X3d_m.shape
                #X3d_m=reshapeX1(X3d_m)
                #X4d=np.swapaxes(X4d,1,2) # or np.transpose(x,(1,0,2))  1,0,2 is axis
                te_X1=X3d_m
                te_X2=X3d_ild
                te_X1 = reshapeX1(te_X1)
                te_X2 = reshapeX3(te_X2)
                
                if debug:
                    # with a Sequential model
                    #md.summary()
                    print na
                    get_3rd_layer_output = K.function([md.layers[0].input, K.learning_phase()], [md.layers[4].output])
                    layer_output = get_3rd_layer_output([te_X1, 0])[0]
                    print layer_output.shape
                    #layer_output1=layer_output[5,:,:]
                    layer_output1=layer_output[:,16,:]
                    imgplot=plt.matshow((layer_output1.T))
                    #imgplot.set_cmap('spectral')
                    plt.colorbar()
                    plt.show()
                    sys.pause()
                
                p_y_pred = md.predict( [te_X1,te_X2] )
                #p_y_pred = md.predict( te_X1 )
                p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
                pred = np.zeros(n_labels)
                pred[ np.where(p_y_pred>thres) ] = 1
                ind=0
                for la in cfg.labels:
                    if la=='S':
                        break
                    elif la=='c':
                        y_true_file_c.append(na)
                        y_true_binary_c.append(y[ind])
                    elif la=='m':
                        y_true_file_m.append(na)
                        y_true_binary_m.append(y[ind])
                    elif la=='f':
                        y_true_file_f.append(na)
                        y_true_binary_f.append(y[ind])
                    elif la=='v':
                        y_true_file_v.append(na)
                        y_true_binary_v.append(y[ind])
                    elif la=='p':
                        y_true_file_p.append(na)
                        y_true_binary_p.append(y[ind])
                    elif la=='b':
                        y_true_file_b.append(na)
                        y_true_binary_b.append(y[ind])
                    elif la=='o':
                        y_true_file_o.append(na)
                        y_true_binary_o.append(y[ind])
                    result=[na,la,p_y_pred[ind]]
                    result_roll.append(result)
                    ind=ind+1
                
                
                pred_roll.append( pred )
                gt_roll.append( y )
    
    pred_roll = np.array( pred_roll )
    gt_roll = np.array( gt_roll )
    #write csv for EER computation
    csvfile=file('result.csv','wb')
    writer=csv.writer(csvfile)
    #writer.writerow(['fn','label','score'])
    writer.writerows(result_roll)
    csvfile.close()
    
    # calculate prec, recall, fvalue
    prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
    # EER for each tag : [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
    EER_c=eer.compute_eer('result.csv', 'c', dict(zip(y_true_file_c, y_true_binary_c)))
    EER_m=eer.compute_eer('result.csv', 'm', dict(zip(y_true_file_m, y_true_binary_m)))
    EER_f=eer.compute_eer('result.csv', 'f', dict(zip(y_true_file_f, y_true_binary_f)))
    EER_v=eer.compute_eer('result.csv', 'v', dict(zip(y_true_file_v, y_true_binary_v)))
    EER_p=eer.compute_eer('result.csv', 'p', dict(zip(y_true_file_p, y_true_binary_p)))
    EER_b=eer.compute_eer('result.csv', 'b', dict(zip(y_true_file_b, y_true_binary_b)))
    EER_o=eer.compute_eer('result.csv', 'o', dict(zip(y_true_file_o, y_true_binary_o)))
    EER=(EER_c+EER_m+EER_v+EER_p+EER_f+EER_b+EER_o)/7.0
    print prec, recall, fvalue
    print EER_c,EER_m,EER_f,EER_v,EER_p,EER_b,EER_o
    print EER

if __name__ == '__main__':
    recognize()
