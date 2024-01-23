import tensorflow
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd

VALIDATION_DATA_DIR = 'data/dense_val_data.csv'
NUM_INPUTS = 4
NUM_CLASSES = 1
BATCH_SIZE = 128


def train():
    df_val = pd.read_csv(VALIDATION_DATA_DIR)
    x_test = np.zeros((len(df_val), NUM_INPUTS))
    y_test = np.zeros((len(df_val), 1))
    model = load_model('models/train/dense/14/model_best.287-0.0106.keras')

    print(model.summary())

    score = 0

    for index, row in df_val.iterrows():
        x_test[index][0] = 1.0 - float(row['div_area'])
        x_test[index][1] = 1.0 - float(row['div_dist'])
        x_test[index][2] = float(row['resnet'])
        x_test[index][3] = float(row['conv_next'])
        y_test[index][0] = float(row['value'])

        expanded_array = np.expand_dims(x_test[index], axis=0)
        pred = model.predict(expanded_array)

        print(row['name'])
        print("expected: " + str(y_test[index][0]))
        print("predict: " + str(pred[0]))

        actual_val = 0
        if pred[0] < 0.5:
            actual_val = 0.
        else:
            actual_val = 1.

        expected_val = y_test[index][0]

        if actual_val == expected_val:
            score = score + 1

    print(float(score) / float(len(df_val)))



if __name__ == '__main__':
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
    train()
