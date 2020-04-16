# keras 模型
from keras.layers import Input, Dense, LSTM as LSTM_keras
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def get_keras_model(config):
    input1 = Input(shape=(config.time_step, config.input_size))
    lstm = LSTM_keras(units=config.hidden_size,dropout=config.dropout_rate,return_sequences=True)(input1)
    output = Dense(config.output_size)(lstm)
    model = Model(input1, output)
    model.compile(loss='mse', optimizer='adam', metrics=["mae"])
    return model

def train(config, train_X, train_Y, valid_X, valid_Y):
    model = get_keras_model(config)
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