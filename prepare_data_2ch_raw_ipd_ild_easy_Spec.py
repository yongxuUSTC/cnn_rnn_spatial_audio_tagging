 
import sys
sys.path.append('../Hat')
from Hat.preprocessing import mat_2d_to_3d, reshape_3d_to_4d, mat_concate_multiinmaps6in
import numpy as np
from scipy import signal
import cPickle
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal
import wavio
import librosa
import config_2ch_raw_spec_ipld as cfg
import csv
import scipy.stats
from sklearn import preprocessing


### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

### def segment raw date
def segraw(x, nperseg, noverlap):
    step = nperseg - noverlap
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,strides=strides)
    return np.array(result)

# calculate mel feature
def GetMel( wav_fd, fe_fd_left, fe_fd_right, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz2ch.wav') ]
    extlen=len('16kHz2ch.wav')+1
    #print extlen
    #sys.exit()
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        #print wav.shape
        #print fs
        #sys.exit()
        if ( wav.ndim==2 ): 
            wav_m = np.mean( wav, axis=-1 ) # mean
            wav_l = wav[:,0]                # left
            wav_r = wav[:,1]                # right
            wav_d = wav_l-wav_r             # difference
            #wavio.write('wav_m.wav',wav_m,16000,sampwidth=2)
            #wavio.write('wav_l.wav',wav_l,16000,sampwidth=2)
            #wavio.write('wav_r.wav',wav_r,16000,sampwidth=2)
            #wavio.write('wav_d.wav',wav_d,16000,sampwidth=2)
        #print wav_m.shape, wav_d.shape, wav_l.shape, wav_r.shape
        #sys.exit()
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        

        #X_m=segraw(wav_m, cfg.win, cfg.win/2)#overlap bigger, more training samples
        #X_r=segraw(wav_r, cfg.win, cfg.win/2)
        #X_d=segraw(wav_d, cfg.win, cfg.win/2)
        #X_l=segraw(wav_l, cfg.win, cfg.win/2)
        X_m=segraw(wav_m, cfg.win, 0)#overlap bigger, more training samples
        X_r=segraw(wav_r, cfg.win, 0)
        X_d=segraw(wav_d, cfg.win, 0)
        X_l=segraw(wav_l, cfg.win, 0)


        #[f_l, t_l, X_l] = signal.spectral.spectrogram( wav_l, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=False, mode='magnitude' )
        [f_l, t_l, X_l] = signal.spectral.spectrogram( wav_l, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=False, mode='magnitude' )
        X_l = X_l.T
        [f_r, t_r, X_r] = signal.spectral.spectrogram( wav_r, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=False, mode='magnitude' )
        X_r = X_r.T
        
        #X_ild=20*np.log10(abs(np.divide(X_l,X_r)))
        X_ipd=np.angle(np.divide(X_l,X_r),deg=0)
        X_ild=X_l-X_r
        print X_ild.shape
        
        #print np.max(X_ild),np.min(X_ild)
        #system.exit()
        # DEBUG. print mel-spectrogram
        #plt.matshow(X_ild, origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path_left = fe_fd_left + '/' + na[0:-extlen] + '.f'
        out_path_right = fe_fd_right + '/' + na[0:-extlen] + '.f'
        out_path_mean = fe_fd_mean + '/' + na[0:-extlen] + '.f'
        out_path_ipd = fe_fd_ipd + '/' + na[0:-extlen] + '.f'
        out_path_ild = fe_fd_ild + '/' + na[0:-extlen] + '.f'
        out_path_diff = fe_fd_diff + '/' + na[0:-extlen] + '.f'
        cPickle.dump( X_l, open(out_path_left, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_r, open(out_path_right, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_m, open(out_path_mean, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_d, open(out_path_diff, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_ipd, open(out_path_ipd, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_ild, open(out_path_ild, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

# calculate mel feature
def GetMBK( wav_fd, fe_fd_left, fe_fd_right, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz2ch.wav') ]
    extlen=len('16kHz2ch.wav')+1
    #print extlen
    #sys.exit()
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        #print wav.shape
        #print fs
        #sys.exit()
        if ( wav.ndim==2 ): 
            wav_m = np.mean( wav, axis=-1 ) # mean
            wav_l = wav[:,0]                # left
            wav_r = wav[:,1]                # right
            #wav_d = wav_l-wav_r             # difference
            #wavio.write('wav_m.wav',wav_m,16000,sampwidth=2)
            #wavio.write('wav_l.wav',wav_l,16000,sampwidth=2)
            #wavio.write('wav_r.wav',wav_r,16000,sampwidth=2)
            #wavio.write('wav_d.wav',wav_d,16000,sampwidth=2)
        #print wav_m.shape, wav_d.shape, wav_l.shape, wav_r.shape
        #sys.exit()
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)

        [f_m, t_m, X_m] = signal.spectral.spectrogram( wav_m, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' )
        X_m = X_m.T
        [f_l, t_l, X_l] = signal.spectral.spectrogram( wav_l, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=False, mode='magnitude' )
        X_l = X_l.T
        [f_r, t_r, X_r] = signal.spectral.spectrogram( wav_r, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=False, mode='magnitude' )
        X_r = X_r.T
        
        #X_ild=20*np.log10(abs(np.divide(X_l,X_r)))
        #X_ipd=np.angle(np.divide(X_l,X_r),deg=0)
        X_ild=X_l-X_r
        print X_ild.shape

        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=40, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X_m = np.dot( X_m, melW.T )
        print X_m.shape

        #print np.max(X_ild),np.min(X_ild)
        #system.exit()
        # DEBUG. print mel-spectrogram
        #plt.matshow(X_ild, origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path_left = fe_fd_left + '/' + na[0:-extlen] + '.f'
        out_path_right = fe_fd_right + '/' + na[0:-extlen] + '.f'
        out_path_mean = fe_fd_mean + '/' + na[0:-extlen] + '.f'
        out_path_ipd = fe_fd_ipd + '/' + na[0:-extlen] + '.f'
        out_path_ild = fe_fd_ild + '/' + na[0:-extlen] + '.f'
        out_path_diff = fe_fd_diff + '/' + na[0:-extlen] + '.f'
        #cPickle.dump( X_l, open(out_path_left, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_r, open(out_path_right, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_m, open(out_path_mean, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_d, open(out_path_diff, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_ipd, open(out_path_ipd, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        cPickle.dump( X_ild, open(out_path_ild, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )

# calculate mel feature
def GetSpec( wav_fd, fe_fd_left, fe_fd_right, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz2ch.wav') ]
    extlen=len('16kHz2ch.wav')+1
    #print extlen
    #sys.exit()
    num=0
    names = sorted(names)
    for na in names:
        num=num+1
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        #print wav.shape
        #print fs
        #sys.exit()
        if ( wav.ndim==2 ): 
            wav_m = np.mean( wav, axis=-1 ) # mean
            wav_l = wav[:,0]                # left
            wav_r = wav[:,1]                # right
            wav_d = wav_l-wav_r             # difference
            #wavio.write('wav_m.wav',wav_m,16000,sampwidth=2)
            #wavio.write('wav_l.wav',wav_l,16000,sampwidth=2)
            #wavio.write('wav_r.wav',wav_r,16000,sampwidth=2)
            #wavio.write('wav_d.wav',wav_d,16000,sampwidth=2)
        #print wav_m.shape, wav_d.shape, wav_l.shape, wav_r.shape
        #sys.exit()
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)

        [f_m, t_m, X_m] = signal.spectral.spectrogram( wav_m, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' )
        X_m = X_m.T


        [f_d, t_d, X_d] = signal.spectral.spectrogram( wav_d, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' )
        X_d = X_d.T

        [f_l, t_l, X_l] = signal.spectral.spectrogram( wav_l, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' )
        X_l = X_l.T
        [f_r, t_r, X_r] = signal.spectral.spectrogram( wav_r, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' )
        X_r = X_r.T

        [f_l1, t_l1, X_l1] = signal.spectral.spectrogram( wav_l, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='complex' )
        X_l1 = X_l1.T
        [f_r1, t_r1, X_r1] = signal.spectral.spectrogram( wav_r, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='complex' )
        X_r1 = X_r1.T

        #X_m = librosa.core.stft( wav_m, n_fft=512, hop_length=cfg.win/2, win_length=512, window=ham_win, center=True)
        #X_m = X_m.T
        #X_m=np.abs(X_m)
        #X_l = librosa.core.stft( wav_l, n_fft=512, hop_length=cfg.win/2, win_length=512, window=ham_win, center=True)
        #X_l = X_l.T
        #X_r = librosa.core.stft( wav_r, n_fft=512, hop_length=cfg.win/2, win_length=512, window=ham_win, center=True)
        #X_r = X_r.T
        
        X_ild1=20*np.log10(abs(np.divide((X_l1),(X_r1))))
        #X_ild1=20*np.log10(((abs(X_l1)/abs(X_r1))))
        #X_ild1=20*np.log10(abs((X_l1/X_r1)))
        X_ipd1=np.angle(np.divide(X_l1,X_r1),deg=0)
        #X_ipd=np.angle((X_l/X_r),deg=0)
        X_ild=X_l-X_r
        print X_ild.shape

        #print np.max(X_ild),np.min(X_ild)
        #system.exit()
        # DEBUG. print mel-spectrogram
        print X_ild
        va=1
        if num == 3:
            fig=plt.figure()
            ax=fig.add_subplot(4,1,1)
            ax.matshow(X_m.T, origin='lower', aspect='auto', vmin=-va,vmax=va)# cmap='gray_r'
            plt.xlabel('Frames')
            plt.ylabel('Frequency bins')
            plt.title('Spectrogram',fontweight='bold',loc='right')
            ax=fig.add_subplot(4,1,2)
            ax.matshow(X_ild.T, origin='lower', aspect='auto', vmin=-va,vmax=va)
            plt.xlabel('Frames')
            plt.ylabel('Frequency bins')
            plt.title('IMD',fontweight='bold',loc='right')
            ax=fig.add_subplot(4,1,3)
            ax.matshow(X_d.T, origin='lower', aspect='auto', vmin=-va,vmax=va)
            plt.xlabel('Frames')
            plt.ylabel('Frequency bins')
            plt.title('ILD',fontweight='bold',loc='right')
            ax=fig.add_subplot(4,1,4)
            ax.matshow(X_ild1.T, origin='lower', aspect='auto', vmin=-va,vmax=va)
            plt.xlabel('Frames')
            plt.ylabel('Frequency bins')
            plt.title('IPD',fontweight='bold',loc='right')
            plt.show()
            pause
        
        out_path_left = fe_fd_left + '/' + na[0:-extlen] + '.f'
        out_path_right = fe_fd_right + '/' + na[0:-extlen] + '.f'
        out_path_mean = fe_fd_mean + '/' + na[0:-extlen] + '.f'
        out_path_ipd = fe_fd_ipd + '/' + na[0:-extlen] + '.f'
        out_path_ild = fe_fd_ild + '/' + na[0:-extlen] + '.f'
        out_path_diff = fe_fd_diff + '/' + na[0:-extlen] + '.f'
        #cPickle.dump( X_l, open(out_path_left, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_r, open(out_path_right, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_m, open(out_path_mean, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_d, open(out_path_diff, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_ipd, open(out_path_ipd, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
        #cPickle.dump( X_ild, open(out_path_ild, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
          
### format label
# get tags
def GetTags( info_path ):
    with open( info_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    tags = lis[-2][1]
    return tags
            
# tags to categorical, shape: (n_labels)
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y

# get chunk data, size: N*agg_num*n_in
def GetAllData( fe_fd_right, fe_fd_left, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path_left = fe_fd_left + '/' + na + '.f'
        fe_path_right = fe_fd_right + '/' + na + '.f'
        fe_path_mean = fe_fd_mean + '/' + na + '.f'
        fe_path_diff = fe_fd_diff + '/' + na + '.f'
        fe_path_ipd = fe_fd_ipd + '/' + na + '.f'
        fe_path_ild = fe_fd_ild + '/' + na + '.f'
        #fe_path_ori = fe_fd_ori + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        #print info_path
        y = TagsToCategory( tags )
        X_l = cPickle.load( open( fe_path_left, 'rb' ) )
        X_r = cPickle.load( open( fe_path_right, 'rb' ) )
        X_m = cPickle.load( open( fe_path_mean, 'rb' ) )
        X_d = cPickle.load( open( fe_path_diff, 'rb' ) )
        X_ipd = cPickle.load( open( fe_path_ipd, 'rb' ) )
        X_ild = cPickle.load( open( fe_path_ild, 'rb' ) )
        #X_o = cPickle.load( open( fe_path_ori, 'rb' ) )
        
        # aggregate data
        #print X_l.shape #(nframe=125,ndim=257)
        X3d_l = mat_2d_to_3d( X_l, agg_num, hop )
        #print X3d_l.shape # (nsampelPERutt=10,contextfr=33,ndim=257)
        X3d_r = mat_2d_to_3d( X_r, agg_num, hop )
        X3d_m = mat_2d_to_3d( X_m, agg_num, hop )
        X3d_d = mat_2d_to_3d( X_d, agg_num, hop )
        X3d_ipd = mat_2d_to_3d( X_ipd, agg_num, hop )
        X3d_ild = mat_2d_to_3d( X_ild, agg_num, hop )
        #X3d_o = mat_2d_to_3d( X_o, agg_num, hop )
        # reshape 3d to 4d
        #X4d_l = reshape_3d_to_4d( X3d_l)
        #X4d_r = reshape_3d_to_4d( X3d_r)
        #X4d_m = reshape_3d_to_4d( X3d_m)
        #X4d_d = reshape_3d_to_4d( X3d_d)
        # concatenate
        X4d=mat_concate_multiinmaps6in(X3d_l, X3d_r, X3d_m, X3d_d, X3d_ipd, X3d_ild)
        #print X4d.shape      
        #sys.exit()       
        
        if curr_fold==fold:
            te_Xlist.append( X4d )
            te_ylist += [ y ] * len( X4d )
        else:
            tr_Xlist.append( X4d )
            tr_ylist += [ y ] * len( X4d )

    return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ),\
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist )


# get chunk data, size: N*agg_num*n_in
def GetAllData_separate( fe_fd_right, fe_fd_left, fe_fd_mean, fe_fd_diff, fe_fd_ipd, fe_fd_ild, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist1, tr_Xlist2, tr_ylist = [], [], []
    te_Xlist1, te_Xlist2, te_ylist = [], [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        #fe_path_left = fe_fd_left + '/' + na + '.f'
        #fe_path_right = fe_fd_right + '/' + na + '.f'
        fe_path_mean = fe_fd_mean + '/' + na + '.f'
        #fe_path_diff = fe_fd_diff + '/' + na + '.f'
        #fe_path_ipd = fe_fd_ipd + '/' + na + '.f'
        fe_path_ild = fe_fd_ild + '/' + na + '.f'
        #fe_path_ori = fe_fd_ori + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        #print info_path
        y = TagsToCategory( tags )
        #X_l = cPickle.load( open( fe_path_left, 'rb' ) )
        #X_r = cPickle.load( open( fe_path_right, 'rb' ) )
        X_m = cPickle.load( open( fe_path_mean, 'rb' ) )
        #X_d = cPickle.load( open( fe_path_diff, 'rb' ) )
        #X_ipd = cPickle.load( open( fe_path_ipd, 'rb' ) )
        X_ild = cPickle.load( open( fe_path_ild, 'rb' ) )
        #X_o = cPickle.load( open( fe_path_ori, 'rb' ) )
        
        # aggregate data
        #print X_l.shape #(nframe=125,ndim=257)
        #X3d_l = mat_2d_to_3d( X_l, agg_num, hop )
        #print X3d_l.shape # (nsampelPERutt=10,contextfr=33,ndim=257)
        #X3d_r = mat_2d_to_3d( X_r, agg_num, hop )
        X3d_m = mat_2d_to_3d( X_m, agg_num, hop )
        #X3d_d = mat_2d_to_3d( X_d, agg_num, hop )
        #X3d_ipd = mat_2d_to_3d( X_ipd, agg_num, hop )
        X3d_ild = mat_2d_to_3d( X_ild, agg_num, hop )
        #X3d_o = mat_2d_to_3d( X_o, agg_num, hop )
        # reshape 3d to 4d
        #X4d_l = reshape_3d_to_4d( X3d_l)
        #X4d_r = reshape_3d_to_4d( X3d_r)
        #X4d_m = reshape_3d_to_4d( X3d_m)
        #X4d_d = reshape_3d_to_4d( X3d_d)
        # concatenate
        #X4d=mat_concate_multiinmaps6in(X3d_l, X3d_r, X3d_m, X3d_d, X3d_ipd, X3d_ild)
        #print X4d.shape      
        #sys.exit()       
        
        if curr_fold==fold:
            te_Xlist1.append( X3d_m )
            te_Xlist2.append( X3d_ild )
            te_ylist += [ y ] * len( X3d_m )
        else:
            tr_Xlist1.append( X3d_m )
            tr_Xlist2.append( X3d_ild )
            tr_ylist += [ y ] * len( X3d_m )

    return np.concatenate( tr_Xlist1, axis=0 ), np.concatenate( tr_Xlist2, axis=0 ), np.array( tr_ylist ),\
           np.concatenate( te_Xlist1, axis=0 ), np.concatenate( te_Xlist2, axis=0 ), np.array( te_ylist )
    
# size: n_songs*n_chunks*agg_num*n_in
def GetSegData( fe_fd, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )    
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ]
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ]

    return np.array( tr_Xlist ), np.array( tr_ylist ), \
           np.array( te_Xlist ), np.array( te_ylist )
           
def GetScaler( fe_fd, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        if curr_fold!=fold:
            tr_Xlist.append( X )
            
    Xall = np.concatenate( tr_Xlist, axis=0 )
    scaler = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( Xall )

    return scaler
           
def GetScalerSegData( fe_fd, agg_num, hop, fold, scaler ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        if scaler is not None:
            X = scaler.transform( X )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ]
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ]

    return np.array( tr_Xlist ), np.array( tr_ylist ), \
           np.array( te_Xlist ), np.array( te_ylist )
    
###
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
            
if __name__ == "__main__":
    CreateFolder( cfg.scrap_fd + '/Fe' )
    CreateFolder( cfg.scrap_fd + '/Fe/Mel_l' )
    CreateFolder( cfg.scrap_fd + '/Fe/Mel_r' )
    CreateFolder( cfg.scrap_fd + '/Fe/Mel_m' )
    CreateFolder( cfg.scrap_fd + '/Fe/Mel_d' )
    CreateFolder( cfg.scrap_fd + '/Fe/Mel_ild' )
    CreateFolder( cfg.scrap_fd + '/Fe/Mel_ipd' )
    CreateFolder( cfg.scrap_fd + '/Results' )
    CreateFolder( cfg.scrap_fd + '/Md' )
    GetSpec( cfg.dev_wav_fd, cfg.dev_fe_mel_fd_left, cfg.dev_fe_mel_fd_right, cfg.dev_fe_mel_fd_mean, cfg.dev_fe_mel_fd_diff, cfg.dev_fe_mel_fd_ipd, cfg.dev_fe_mel_fd_ild, n_delete=0 )
