import keras
from keras import Input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import resnet50
import numpy as np
import os

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_model():
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    # create the base pre-trained model
    base_model = resnet50.ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        Dense(2, activation='softmax')
    ])
    return model


def get_data():

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    train_dir = os.path.join(PATH, 'all')
    train_rail_dir = os.path.join(train_dir, 'rail')
    train_nonrail_dir = os.path.join(train_dir, 'nonrail')
    train_rail_size = len(os.listdir(train_rail_dir))
    train_nonrail_size = len(os.listdir(train_nonrail_dir))


    train_images = []

    for i in tqdm(os.listdir(train_rail_dir)):
        img_path = os.path.join(train_rail_dir, i)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        train_images.append([np.array(img), [1, 0]])

    for i in tqdm(os.listdir(train_nonrail_dir)):
        img_path = os.path.join(train_nonrail_dir, i)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        train_images.append([np.array(img), [0, 1]])

    train_img = np.array([i[0] for i in train_images]).reshape(-1, 224, 224, 3)
    train_label = np.array([i[1] for i in train_images])
    return train_img, train_label

if __name__ == "__main__":
    resnet_model = get_model()
    train_img, train_label = get_data()

    init_epoch = 10
    lr = 1e-3

    resnet_model.compile(loss=keras.losses.categorical_crossentropy,
                         optimizer=keras.optimizers.Adam(lr=lr),
                         metrics=['accuracy'])


    resnet_model.fit(x=train_img, y=train_label, batch_size=64, epochs=init_epoch, verbose=1, callbacks=None, validation_split=0.1)

    resnet_model.summary()

    '''
    model_json = resnet_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(resnet_model)
    # serialize weights to HDF5
    resnet_model.save_weights("model.h5")
    print("Saved model to disk")
    '''