import argparse
import os

import numpy as np
import torch.cuda

def parse_args():
    parser = argparse.ArgumentParser(description="Add args to the model")
    parser.add_argument("--condition", type=str, default='subject_dependency', help="The condition of the experiment")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    dir_name = './SEED-IV'
    sessions = os.listdir(dir_name)
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    people_num = 15
    class_num = 4
    # 读取文件夹中的所有数据
    for session in sessions:
        session_dir = dir_name + '/' + session
        peoples = os.listdir(session_dir)
        for people in peoples:
            people_dir = session_dir + '/' + people + '/'
            train_data = np.load(people_dir + 'train_data.npy')
            train_label = np.load(people_dir + 'train_label.npy')
            test_data = np.load(people_dir + 'test_data.npy')
            test_label = np.load(people_dir + 'test_label.npy')
            train_data = train_data.reshape((train_data.shape[0], -1))
            test_data = test_data.reshape((test_data.shape[0], -1))
            train_datas.append(train_data)
            train_labels.append(train_label)
            test_datas.append(test_data)
            test_labels.append(test_label)
