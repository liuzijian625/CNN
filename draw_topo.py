import mne
import matplotlib.pyplot as plt

def draw_topo(loc_path, emotionList, emotionData, bandList, titleLoc,savepath):
    """画脑壳图

    :param str loc_path: 包含'channel_62_pos.locs'字段的位置。
    :param list emotionList: 情绪名称组成的列表，图中情绪的顺序与列表中的顺序一致
    :param Dict emotionData: 每种情绪的数据。
        key为情绪的序号，从0开始，编号顺序与emotionList中情绪的的顺序一致。
        value的格式为：62*5。62为导联数，5为频段数。
        注：所有被试的数据需要先归一化后再对每种情绪做平均，最终一种情绪的数据维度只有62*5
    :param list bandList: 5个频段名称组成的列表。频段顺序需要与emotionData的value中顺序一致
    :param float titleLoc: 调整title的距离
    :param str savepath: 包含图片名称、后缀的图片保存路径
    """

    # montage = mne.channels.make_standard_montage('standard_1020')
    montage = mne.channels.read_custom_montage(loc_path)

    ch_list = ['Fp1','Fpz','Fp2','AF3','AF4','F7','F5','F3','F1','Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1',
'FCz','FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3',
'CP1','CPz','CP2','CP4','CP6','TP8','P7','P5','P3','P1','Pz','P2','P4','P6','P8','PO7','PO5','PO3',
'POz','PO4','PO6','PO8','CB1','O1','Oz','O2','CB2']

    info = mne.create_info(ch_names=ch_list, sfreq=200., ch_types='eeg', verbose=None)
    print(info)

    fig, axes = plt.subplots(len(emotionList), len(bandList), figsize=(15, 15))
    for emotion_index, emotion in enumerate(emotionList):
        axes[emotion_index][0].get_yaxis().set_label_coords(-0.2, 0.38)
        axes[emotion_index][0].set_ylabel(emotion, fontdict={'fontsize': 30}, rotation=90, labelpad=4)
        for band_index in range(len(bandList)):
            # Only plot the x label in last row
            if emotion_index == len(Emotion) - 1:
                axes[emotion_index][band_index].set_xlabel(band_list[band_index], fontdict={'fontsize': 30},
                                                           labelpad=100)
            # plt.axes(axes[emotion_index][band_index])

            Data = emotionData[str(emotion_index)]
            evoked = mne.EvokedArray(np.array(Data)[:,band_index].reshape(62,1), info)
            evoked.set_montage(montage)
            x, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, cmap='seismic',axes=axes[emotion_index][band_index],
                                        show=False, sensors=False, outlines='head')
            plt.tight_layout()
    cb = plt.colorbar(x, ax=axes, shrink=0.6, pad=0.05)
    cb.ax.tick_params(labelsize='25')
    # gender = savepath.split('_')[-1].split('.')[0]
    plt.suptitle(f'{Dataset}', y=titleLoc, fontsize=30)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()


if __name__=='__main__':
    """
    .locs文件为EEGLAB提供的电极坐标位置，每列代表channum, theta, radius, labels 
    mne可以通过.locs文件获取电极的3维坐标
    mne存储方式为：Montages contain sensor positions in 3D (x, y, z in meters)
    """
    montage = mne.channels.read_custom_montage('channel_62_pos.locs')
    a = montage.get_positions()
    ch_pos = a['ch_pos']    # sensor positions in 3D (x, y, z in meters)
    montage.plot()
    plt.show()
