import tensorflow
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

TRAIN_DATA_DIR = 'data/dense_train_data.csv'
VALIDATION_DATA_DIR = 'data/dense_val_data.csv'
NUM_INPUTS = 4
NUM_CLASSES = 1
BATCH_SIZE = 128


def train():
    df_train = pd.read_csv(TRAIN_DATA_DIR)
    df_val = pd.read_csv(VALIDATION_DATA_DIR)
    x_train = np.zeros((len(df_train) * 2, NUM_INPUTS))
    y_train = np.zeros((len(df_train) * 2, 1))
    x_test = np.zeros((len(df_val) * 2, NUM_INPUTS))
    y_test = np.zeros((len(df_val) * 2, 1))

    train_index = 0
    val_index = 0

    for index, row in df_train.iterrows():
        x_train[train_index][0] = 1.0 - float(row['div_area'])
        x_train[train_index][1] = 1.0 - float(row['div_dist'])
        x_train[train_index][2] = float(row['resnet'])
        x_train[train_index][3] = float(row['conv_next'])
        y_train[train_index][0] = float(row['value'])
        train_index = train_index + 1

    for index, row in df_val.iterrows():
        x_test[val_index][0] = 1.0 - float(row['div_area'])
        x_test[val_index][1] = 1.0 - float(row['div_dist'])
        x_test[val_index][2] = float(row['resnet'])
        x_test[val_index][3] = float(row['conv_next'])
        y_test[val_index][0] = float(row['value'])
        val_index = val_index + 1

    model = Sequential()
    model.add(Dense(256, input_dim=NUM_INPUTS, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    # model = load_model('models/dense_21/model_best.321-0.1731.keras')
    #
    # print(model.summary())

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001),
        loss='MAE')

    dir_path = "models/dense/01/"
    os.makedirs(dir_path)
    file_name = "model_best.{epoch:02d}-{val_loss:.4f}.keras"
    filepath = dir_path + file_name

    earlystopper = EarlyStopping(patience=100, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=800,
                        callbacks=callbacks_list, shuffle=True)

    model.save(dir_path + 'model_last.keras')

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(dir_path + 'out.csv', index=False)

    plt.figure(figsize=(15, 15))
    plt.plot(history.history['loss'],
             label='Показатель ошибок на обучающем наборе')
    plt.plot(history.history['val_loss'],
             label='Показатель ошибок на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Показатель ошибок')
    plt.legend()
    plt.savefig(dir_path + 'loss.png')


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    train()
