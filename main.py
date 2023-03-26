import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Add args to the model")
    parser.add_argument("--condition", type=str, default='subject_independency', help="The condition of the experiment")
    parser.add_argument("--data_type", type=str, default='1D_not_reshape', help="Do you want to reshape the feature")
    parser.add_argument("--use_GPU", type=bool, default=True, help="Do you want to use GPU to train")
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--shuffle", type=bool, default=True, help="whether shuffle the train data")
    parser.add_argument("--drop_last", type=bool, default=False, help="whether drop the last train data")
    parser.add_argument("--num_workers", type=int, default=0, help="number of the workers")
    parser.add_argument("--is_norm", type=bool, default=True, help="whether normalize the data")
    parser.add_argument("--loop_times", type=int, default=100, help="the number of loops")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--lr", type=float, default=0.001, help="learn_rate")
    parser.add_argument("--people_num", type=int, default=15, help="the number of people")
    parser.add_argument("--class_num", type=int, default=4, help="the number of class")
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self, data, label, use_GPU, is_norm, mean_data, var_data, data_type):
        self.data = data
        self.label = label
        self.use_GPU = use_GPU
        self.is_norm = is_norm
        self.mean_data = mean_data
        self.var_data = var_data
        self.data_type = data_type

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        if self.is_norm:
            data = (data - self.mean_data) / np.sqrt(self.var_data)
        label = torch.LongTensor([self.label[index]])
        if self.use_GPU and torch.cuda.is_available():
            data = torch.cuda.FloatTensor(self.data[index])
            label = torch.cuda.LongTensor([self.label[index]])
            data = data.cuda()
            label = label.cuda()
        if self.data_type == '1D_not_reshape':
            return data, label
        return data.unsqueeze(0), label

    def __len__(self):
        return len(self.data)


class CNN2d(nn.Module):
    def __init__(self, class_num, data_shape):
        super().__init__()
        # conv1
        self.data_shape = data_shape
        self.in_channels_conv1 = 1
        self.out_channels_conv1 = 32
        self.kernel_size_conv1 = 3
        self.stride_conv1 = 1
        self.padding_conv1 = 1
        self.padding_mode_conv1 = 'zeros'
        self.bias_conv1 = True
        # pool
        self.kernel_size_pool = 2
        self.stride_pool = self.kernel_size_pool
        self.padding_pool = 0
        # conv2
        self.in_channels_conv2 = self.out_channels_conv1
        self.out_channels_conv2 = 64
        self.kernel_size_conv2 = 3
        self.stride_conv2 = 1
        self.padding_conv2 = 1
        self.padding_mode_conv2 = 'zeros'
        self.bias_conv2 = True
        # fc1
        self.in_features_fc1 = self.out_channels_conv2 * math.floor(
            math.floor(self.data_shape[0] / self.kernel_size_pool) / self.kernel_size_pool) * math.floor(
            math.floor(self.data_shape[1] / self.kernel_size_pool) / self.kernel_size_pool)
        self.out_features_fc1 = 128
        self.bias_fc1 = True
        # fc2
        self.in_features_fc2 = self.out_features_fc1
        self.out_features_fc2 = 64
        self.bias_fc2 = True
        # fc3
        self.in_features_fc3 = self.out_features_fc2
        self.out_features_fc3 = class_num
        self.bias_fc3 = True
        self.conv1 = nn.Conv2d(in_channels=self.in_channels_conv1, out_channels=self.out_channels_conv1,
                               kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1,
                               padding_mode=self.padding_mode_conv1, bias=self.bias_conv1)
        self.pool = nn.MaxPool2d(kernel_size=self.kernel_size_pool, stride=self.stride_pool, padding=self.padding_pool)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels_conv2, out_channels=self.out_channels_conv2,
                               kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2,
                               padding_mode=self.padding_mode_conv2, bias=self.bias_conv2)
        self.fc1 = nn.Linear(in_features=self.in_features_fc1, out_features=self.out_features_fc1, bias=self.bias_fc1)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=self.in_features_fc2, out_features=self.out_features_fc2, bias=self.bias_fc2)
        self.fc3 = nn.Linear(in_features=self.in_features_fc3, out_features=self.out_features_fc3, bias=self.bias_fc3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x=self.drop(x)
        x = F.relu(self.fc2(x))
        # x=self.drop(x)
        x = self.fc3(x)
        return x


class CNN1dNoReshape(nn.Module):
    def __init__(self, class_num, data_shape):
        super().__init__()
        # conv1
        self.data_shape = data_shape
        self.in_channels_conv1 = 5
        self.out_channels_conv1 = 32
        self.kernel_size_conv1 = 3
        self.stride_conv1 = 1
        self.padding_conv1 = 1
        self.padding_mode_conv1 = 'zeros'
        self.bias_conv1 = True
        # pool
        self.kernel_size_pool = 2
        self.stride_pool = self.kernel_size_pool
        self.padding_pool = 0
        # conv2
        self.in_channels_conv2 = self.out_channels_conv1
        self.out_channels_conv2 = 64
        self.kernel_size_conv2 = 3
        self.stride_conv2 = 1
        self.padding_conv2 = 1
        self.padding_mode_conv2 = 'zeros'
        self.bias_conv2 = True
        # fc1
        self.in_features_fc1 = self.out_channels_conv2 * math.floor(
            math.floor(self.data_shape[1] / self.kernel_size_pool) / self.kernel_size_pool)
        self.out_features_fc1 = 128
        self.bias_fc1 = True
        # fc2
        self.in_features_fc2 = self.out_features_fc1
        self.out_features_fc2 = 64
        self.bias_fc2 = True
        # fc3
        self.in_features_fc3 = self.out_features_fc2
        self.out_features_fc3 = class_num
        self.bias_fc3 = True
        self.conv1 = nn.Conv1d(in_channels=self.in_channels_conv1, out_channels=self.out_channels_conv1,
                               kernel_size=self.kernel_size_conv1, stride=self.stride_conv1, padding=self.padding_conv1,
                               padding_mode=self.padding_mode_conv1, bias=self.bias_conv1)
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size_pool, stride=self.stride_pool, padding=self.padding_pool)
        self.conv2 = nn.Conv1d(in_channels=self.in_channels_conv2, out_channels=self.out_channels_conv2,
                               kernel_size=self.kernel_size_conv2, stride=self.stride_conv2, padding=self.padding_conv2,
                               padding_mode=self.padding_mode_conv2, bias=self.bias_conv2)
        self.fc1 = nn.Linear(in_features=self.in_features_fc1, out_features=self.out_features_fc1, bias=self.bias_fc1)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=self.in_features_fc2, out_features=self.out_features_fc2, bias=self.bias_fc2)
        self.fc3 = nn.Linear(in_features=self.in_features_fc3, out_features=self.out_features_fc3, bias=self.bias_fc3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


def data_to_people(people_num, datas, labels):
    """拼接数据，获取每个人对应的数据，用于被试独立"""
    peoples_data_list_version = []
    peoples_label_list_version = []
    for i in range(len(datas)):
        if i < people_num:
            peoples_data_list_version.append([datas[i]])
            peoples_label_list_version.append([labels[i]])
        else:
            peoples_data_list_version[i % 15].append(datas[i])
            peoples_label_list_version[i % 15].append(labels[i])
    peoples_data = []
    peoples_label = []
    for i in range(people_num):
        people_data = []
        people_label = []
        for j in range(len(peoples_data_list_version[i])):
            for k in range(len(peoples_data_list_version[i][j])):
                people_data.append(peoples_data_list_version[i][j][k])
                people_label.append(peoples_label_list_version[i][j][k])
        peoples_data.append(people_data)
        peoples_label.append(people_label)
    return peoples_data, peoples_label


def get_datas(test_num, peoples_datas, peoples_labels):
    """根据轮数，获取对应轮次的训练集和测试集，用于被试独立"""
    train_datas = []
    train_labels = []
    for people in range(len(peoples_datas)):
        if people == test_num:
            test_datas = [peoples_datas[people]]
            test_labels = [peoples_labels[people]]
        else:
            for i in range(len(peoples_datas[people])):
                train_datas.append(peoples_datas[people][i])
                train_labels.append(peoples_labels[people][i])
    return [train_datas], [train_labels], test_datas, test_labels


def train_and_test(train_data, train_label, test_data, test_label, batch_size, shuffle, num_workers, drop_last,
                   class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type):
    mean_data = train_data.mean()
    var_data = train_data.var()
    data_shape = train_data[0].shape
    train_dataset = MyDataset(train_data, train_label, use_GPU, is_norm, mean_data, var_data, data_type)
    test_dataset = MyDataset(test_data, test_label, use_GPU, is_norm, mean_data, var_data, data_type)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_workers, drop_last=drop_last)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=shuffle,
                                  num_workers=num_workers, drop_last=drop_last)
    if data_type == '1D_not_reshape':
        CNN = CNN1dNoReshape(class_num, data_shape)
    elif data_type == '2D':
        CNN = CNN2d(class_num, data_shape)
    if use_GPU:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        CNN.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CNN.parameters(), lr=lr, momentum=momentum)
    CNN.train()
    for epoch in range(loop_times):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            inputs, labels = data
            # print(inputs.shape)
            optimizer.zero_grad()
            outputs = CNN(inputs)
            # print(outputs)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            # loss = torch.nn.NLLLoss()(torch.nn.LogSoftmax()(outputs), labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            outputs1 = CNN(inputs)
            # running_loss += loss.item()
    total_num = 0
    correct_num = 0
    # print(CNN.conv1.weight)
    CNN.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_loader, 0):
            inputs, labels = data
            outputs = CNN(inputs)
            # print(outputs)
            _, predict = torch.max(outputs, 1)
            total_num = total_num + predict.shape[0]
            correct_num = correct_num + (predict == labels.squeeze(1)).sum()
        acc = correct_num / total_num
    return acc.item()


def subject_dependency(train_datas, train_labels, test_datas, test_labels, batch_size, shuffle, num_workers,
                       drop_last, class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type):
    accs = []
    CNN_num = len(train_datas)
    progress_bar = tqdm(range(CNN_num))
    progress_bar.set_description("Steps")
    for i in range(CNN_num):
        progress_bar.update(1)
        acc = train_and_test(train_datas[i], train_labels[i], test_datas[i], test_labels[i], batch_size, shuffle,
                             num_workers, drop_last, class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type)
        accs.append(acc)
        logs = {"acc": acc}
        progress_bar.set_postfix(**logs)
    acc_avg = np.array(accs).sum() / CNN_num
    print("average accuracy:" + str(acc_avg * 100) + "%")


def subject_independency(train_datas, train_labels, test_datas, test_labels, people_num, batch_size, shuffle,
                         num_workers, drop_last, class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type):
    accs = []
    progress_bar = tqdm(range(people_num))
    progress_bar.set_description("Steps")
    peoples_data, peoples_label = data_to_people(people_num, train_datas + test_datas, train_labels + test_labels)
    for i in range(people_num):
        train_datas, train_labels, test_datas, test_labels = get_datas(i, peoples_data, peoples_label)
        train_datas=np.array(train_datas)[0]
        train_labels=np.array(train_labels)[0]
        test_datas=np.array(train_datas)[0]
        test_labels=np.array(test_labels)[0]
        acc = train_and_test(train_datas, train_labels, test_datas, test_labels, batch_size, shuffle,
                             num_workers, drop_last, class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type)
        accs.append(acc)
        logs = {"acc": acc}
        progress_bar.set_postfix(**logs)
    acc_avg = np.array(accs).sum() / people_num
    print("average accuracy:" + str(acc_avg * 100) + "%")


if __name__ == '__main__':
    args = parse_args()
    condition = args.condition
    batch_size = args.batch_size
    shuffle = args.shuffle
    num_workers = args.num_workers
    drop_last = args.drop_last
    data_type = args.data_type
    lr = args.lr
    dir_name = './SEED-IV'
    sessions = os.listdir(dir_name)
    train_datas = []
    train_labels = []
    test_datas = []
    test_labels = []
    people_num = args.people_num
    class_num = args.class_num
    momentum = args.momentum
    loop_times = args.loop_times
    use_GPU = args.use_GPU
    is_norm = args.is_norm
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
            if data_type == '1D_not_reshape':
                train_data = train_data.transpose(0, 2, 1)
                test_data = test_data.transpose(0, 2, 1)
            train_datas.append(train_data)
            train_labels.append(train_label)
            test_datas.append(test_data)
            test_labels.append(test_label)
    if condition == 'subject_dependency':
        subject_dependency(train_datas, train_labels, test_datas, test_labels, batch_size, shuffle,
                           num_workers, drop_last, class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type)
    elif condition == 'subject_independency':
        subject_independency(train_datas, train_labels, test_datas, test_labels, people_num, batch_size, shuffle,
                             num_workers, drop_last, class_num, lr, momentum, loop_times, use_GPU, is_norm, data_type)
