import os

import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('Agg')
import skimage.io as sk


def numpy_to_mne(datas):
    ch_names_df = pd.read_excel('Channel Order.xlsx', header=None, sheet_name='Sheet1', usecols=[0])
    ch_names = ch_names_df.values.tolist()
    ch_names = np.array(ch_names)
    ch_names = ch_names.squeeze(1)
    ch_names = ch_names.tolist()
    ch_types = []
    for i in range(len(ch_names)):
        ch_types.append('eeg')
    sfreq = 100
    new_datas = []
    montage = mne.channels.read_custom_montage("channel_62_pos.locs")
    info = mne.create_info(ch_names, sfreq, ch_types)
    for data in datas:
        raw = mne.io.RawArray(data, info)
        raw.set_montage(montage, on_missing='ignore')
        ica = mne.preprocessing.ICA(n_components=5, random_state=97, max_iter='auto')
        ica.fit(inst=raw)
        ica.plot_components()
        plt.savefig('./temp.jpg')
        new_data = sk.imread('./temp.jpg')
        new_data = new_data.transpose(2, 0, 1)
        new_datas.append(new_data)
    return new_datas

if __name__ == '__main__':
    dir_name = './SEED-IV'
    sessions = os.listdir(dir_name)
    train_datas = []
    test_datas = []
    # 读取文件夹中的所有数据
    for session in sessions:
        session_dir = dir_name + '/' + session
        peoples = os.listdir(session_dir)
        for people in peoples:
            people_dir = session_dir + '/' + people + '/'
            train_data = np.load(people_dir + 'train_data.npy')
            test_data = np.load(people_dir + 'test_data.npy')
            train_data=numpy_to_mne(train_data)
            test_data=numpy_to_mne(test_data)
            train_datas.append(train_data)
            test_datas.append(test_data)
    train_datas=np.array(train_datas)
    test_datas=np.array(test_datas)
    np.save("train_mne.npy",train_datas)
    np.save("test_mne.npy",test_datas)