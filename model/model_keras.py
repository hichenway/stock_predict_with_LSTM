# -*- coding: UTF-8 -*-
"""
@author: hichenway
@知乎: 海晨威
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
keras 模型
"""

from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def get_keras_model(config):
    input1 = Input(shape=(config.time_step, config.input_size))
    lstm = input1
    for i in range(config.lstm_layers):
        lstm = LSTM(units=config.hidden_size,dropout=config.dropout_rate,return_sequences=True)(lstm)
    output = Dense(config.output_size)(lstm)
    model = Model(input1, output)
    model.compile(loss='mse', optimizer='adam')     # metrics=["mae"]
    return model

def gpu_train_init():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7 #最多使用70%GPU内存
    sess_config.gpu_options.allow_growth=True   #初始化时不全部占满GPU显存, 按需分配 
    sess = tf.Session(config = sess_config)
    set_session(sess)

def train(config, logger, train_and_valid_data):
    if config.use_cuda: gpu_train_init()
    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    model = get_keras_model(config)
    model.summary()
    if config.add_train:
        model.load_weights(config.model_save_path + config.model_name)

    check_point = ModelCheckpoint(filepath=config.model_save_path + config.model_name, monitor='val_loss',
                                    save_best_only=True, mode='auto')
    early_stop = EarlyStopping(monitor='val_loss', patience=config.patience, mode='auto')
    model.fit(train_X, train_Y, batch_size=config.batch_size, epochs=config.epoch, verbose=2,
              validation_data=(valid_X, valid_Y), callbacks=[check_point, early_stop])

def predict(config, test_X):
    model = get_keras_model(config)
    model.load_weights(config.model_save_path + config.model_name)
    result = model.predict(test_X, batch_size=1)
    result = result.reshape((-1, config.output_size))
    return result
