# tensorflow 模型
# Author: hichenway
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, config):
        self.config = config

        self.placeholders()
        self.net()
        self.operate()


    def placeholders(self):
        self.X = tf.placeholder(tf.float32, [None,self.config.time_step,self.config.input_size])
        self.Y = tf.placeholder(tf.float32, [None,self.config.time_step,self.config.output_size])


    def net(self):

        def dropout_cell():
            basicLstm = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size)
            dropoutLstm = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=1-self.config.dropout_rate)
            return dropoutLstm

        cell = tf.nn.rnn_cell.MultiRNNCell([dropout_cell() for _ in range(self.config.lstm_layers)])

        output_rnn, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.X, dtype=tf.float32)

        # shape of output_rnn is: [batch_size, time_step, hidden_size]
        self.pred = tf.layers.dense(inputs=output_rnn, units=self.config.output_size)


    def operate(self):
        self.loss = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.Y, [-1])))
        self.optim = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver(tf.global_variables())


def train(config, train_X, train_Y, valid_X, valid_Y):
    with tf.variable_scope("stock_predict"):
        model = Model(config)

    train_len = len(train_X)
    valid_len = len(valid_X)

    if config.use_cuda:
        sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True) #在CUDA可用时会自动选择GPU，否则CPU
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 显存占用率
        sess_config.gpu_options.allow_growth=True   #初始化时不全部占满GPU显存, 按需分配
    else:
        sess_config = None

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        valid_loss_min = float("inf")
        bad_epoch = 0
        for epoch in range(config.epoch):
            print("Epoch {}/{}".format(epoch, config.epoch))
            # 训练
            train_loss_array = []
            for step in range(train_len//config.batch_size):
                feed_dict = {model.X: train_X[step*config.batch_size : (step+1)*config.batch_size],
                             model.Y: train_Y[step*config.batch_size : (step+1)*config.batch_size]}
                train_loss, _ = sess.run([model.loss,model.optim], feed_dict=feed_dict)
                train_loss_array.append(train_loss)

            # 验证与早停
            valid_loss_array = []
            for step in range(valid_len//config.batch_size):
                feed_dict = {model.X: valid_X[step * config.batch_size: (step + 1) * config.batch_size],
                             model.Y: valid_Y[step * config.batch_size: (step + 1) * config.batch_size]}
                valid_loss = sess.run(model.loss, feed_dict=feed_dict)
                valid_loss_array.append(valid_loss)

            valid_loss_cur = np.mean(valid_loss_array)
            print("The train loss is {:.4f}. ".format(np.mean(train_loss_array)),
                  "The valid loss is {:.4f}.".format(valid_loss_cur))

            if valid_loss_cur < valid_loss_min:
                valid_loss_min = valid_loss_cur
                bad_epoch = 0
                path = model.saver.save(sess, config.model_save_path + config.model_name)
                print(path)
            else:
                bad_epoch += 1
                if bad_epoch >= config.patience:
                    print(" The training stops early in epoch {}".format(epoch))
                    break


def predict(config, test_X):
    config.dropout_rate = 0     # 预测模式要调为1

    with tf.variable_scope("stock_predict", reuse=tf.AUTO_REUSE):
        model = Model(config)

    test_len = len(test_X)
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(config.model_save_path)
        model.saver.restore(sess, module_file)

        result = np.zeros((test_len*config.time_step, config.output_size))
        for step in range(test_len):
            feed_dict = {model.X: test_X[step : (step + 1)]}
            test_pred = sess.run(model.pred, feed_dict=feed_dict)
            result[step*config.time_step : (step + 1)*config.time_step] = test_pred[0,:,:]
    return result


