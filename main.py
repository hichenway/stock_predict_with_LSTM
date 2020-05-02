# Author: hichenway
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from log.log_decorator import log

frame = "pytorch"  # 可选： "keras", "pytorch", "tensorflow"
if frame == "pytorch":
    from model.model_pytorch import train, predict
elif frame == "keras":
    from model.model_keras import train, predict
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict
else:
    raise Exception("Wrong frame seletion")

class Config:
    # 数据参数
    feature_columns = list(range(2, 9))     # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    label_columns = [4, 5]                  # 要预测的列，按原数据从0开始计算, 如同时预测第四，五列 最低价和最高价
    # label_in_feature_index = [feature_columns.index(i) for i in label_columns]  # 这样写不行
    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始

    predict_day = 1             # 预测未来几天


    # 网络参数
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 128           # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 2             # LSTM的堆叠层数
    dropout_rate = 0.2          # dropout概率
    time_step = 20              # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False           # 是否载入已有模型参数进行增量训练
    shuffle_train_data = True   # 是否对训练数据做shuffle
    use_cuda = False            # 是否使用GPU训练

    train_data_rate = 0.95      # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.15      # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.001
    epoch = 20                  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42            # 随机种子，保证可复现

    do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    continue_flag = ""           # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 1
        continue_flag = "continue_"

    # 路径参数
    train_data_path = "./data/stock_data.csv"
    model_save_path = "./checkpoint/"
    figure_save_path = "./figure/"
    do_figure_save = False
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

    # 框架参数
    used_frame = frame      # 选择的深度学习框架，不同的框架模型保存后缀不一样
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)              # 数据的均值和方差
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # 归一化，去量纲

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据
        init_data = pd.read_csv(self.config.train_data_path,
                                usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() 是获取列名

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[self.config.predict_day : self.config.predict_day + self.train_num,
                                    self.config.label_in_feature_index]    # 将延后几天的数据作为label

        if not self.config.do_continue_train:
            # 在非连续训练模式下，每time_step行数据会作为一个样本，两个样本错开一行，比如：1-20行，2-21行。。。。
            train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # 在连续训练模式下，每time_step行数据会作为一个样本，两个样本错开time_step行，
            # 比如：1-20行，21-40行。。。到数据末尾，然后又是 2-21行，22-41行。。。到数据末尾，……
            # 这样才可以把上一个样本的final_state作为下一个样本的init_state，而且不能shuffle
            # 目前本项目中仅能在pytorch的RNN系列模型中用
            train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]

        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)   # 划分训练和验证集，并打乱
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        self.start_num_in_test = feature_data.shape[0] % self.config.time_step  # 这些天的数据不够一个time_step
        time_step_size = feature_data.shape[0] // self.config.time_step

        # 在测试数据中，每time_step行数据会作为一个样本，两个样本错开time_step行
        # 比如：1-20行，21-40行。。。到数据末尾。
        test_x = [feature_data[self.start_num_in_test+i*self.config.time_step : self.start_num_in_test+(i+1)*self.config.time_step]
                   for i in range(time_step_size)]
        if return_label_data:       # 实际应用中的测试集是没有label数据的
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)


def draw(config: Config, origin_data: Data, predict_norm_data: np.ndarray):
    label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    assert label_norm_data.shape[0]==predict_norm_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_day天的数据的
    loss = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day] ) ** 2, axis=0)
    print("The mean squared error of stock {} is ".format(label_name), loss)

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    # 通过保存的均值和方差还原数据
    label_data = label_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]

    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]

    if not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            plt.figure(i+1)                     # 预测数据绘制
            plt.plot(label_X, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i], label='predict')
            plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
            print("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day),
                  np.squeeze(predict_data[-config.predict_day:, i]))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

        plt.show()

@log()      # 日志记录装饰器，可以不用改源代码就增加想要的功能
# @log(filename="./log/out.log", to_file=True)      # 如果要记录到文件用这个
def main(config):
    np.random.seed(config.random_seed)              # 设置随机种子，保证可复现
    data_gainer = Data(config)

    if config.do_train:
        train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
        train(config, train_X, train_Y, valid_X, valid_Y)

    if config.do_predict:
        test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
        pred_result = predict(config, test_X)       # 这里输出的是未还原的归一化预测数据
        draw(config, data_gainer, pred_result)


if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)
