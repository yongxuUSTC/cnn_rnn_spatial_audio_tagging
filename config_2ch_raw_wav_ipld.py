'''
SUMMARY:  config file
AUTHOR:   Qiuqiang Kong
Created:  2016.06.23
Modified: YONG XU, 2016.08.09 for multi-channel
--------------------------------------
'''

# development
dev_root = '/vol/vssp/datasets/audio/dcase2016/chime_home'
dev_wav_fd = dev_root + '/chunks_16k_2ch'

# temporary data folder
scrap_fd = "/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_2ch_wav20ms_ipd_ild_overlap"
dev_fe_mel_fd_left = scrap_fd + '/Fe/Mel_l'
dev_fe_mel_fd_right = scrap_fd + '/Fe/Mel_r'
dev_fe_mel_fd_mean = scrap_fd + '/Fe/Mel_m'
dev_fe_mel_fd_diff = scrap_fd + '/Fe/Mel_d'
dev_fe_mel_fd_ipd = scrap_fd + '/Fe/Mel_ipd'
dev_fe_mel_fd_ild = scrap_fd + '/Fe/Mel_ild'
#dev_cv_csv_path = dev_root + '/development_chunks_refined_crossval_dcase2016.csv'
dev_cv_csv_path = dev_root + '/development_chunks_raw_crossval_dcase2016.csv'

# evaluation
'''
eva_csv_path = root + '/evaluation_chunks_refined.csv'
fe_mel_eva_fd = 'Fe_eva/Mel'
'''

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.
win = 320.
