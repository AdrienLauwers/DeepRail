import keras
from keras import Input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten
from keras.applications import resnet50
from keras.models import Model, Sequential
from keras.callbacks import *
import numpy as np
import random
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import matplotlib.image as mpimg
import time


def get_model():
    input_tensor = Input(shape=(HEIGHT, WIDTH, 3))

    # create the base pre-trained model #'imagenet'
    base_model = resnet50.ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    fc_layers = [1024, 1024]
    dropout = 0.5

    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(2, activation='softmax')(x) 

    # model = keras.Sequential([
    #     base_model,
    #     keras.layers.GlobalAveragePooling2D(),
    # ])

    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model

def get_data():
    # PATH = '/content/DeepRailDataset'
    PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PATH = os.path.join(PATH, "DeepRailDataset")
    # train_dir = os.path.join(PATH, 'all')
    train_dir = os.path.join(PATH, 'all')
    train_rail_dir = os.path.join(train_dir, 'rail')
    train_nonrail_dir = os.path.join(train_dir, 'nonrail')
    # train_rail_size = len(os.listdir(train_rail_dir))
    # train_nonrail_size = len(os.listdir(train_nonrail_dir))

    train_images = []

    for i in tqdm(os.listdir(train_rail_dir)):
        img_path = os.path.join(train_rail_dir, i)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (HEIGHT, WIDTH))
        img_rotated = rotate_image(img) # Data augmentation by rotation

        train_images.append([np.array(img).reshape(HEIGHT, WIDTH, 3), [1, 0]])
        train_images.append([np.array(img_rotated), [1, 0]])
    
    for i in tqdm(os.listdir(train_nonrail_dir)):
        img_path = os.path.join(train_nonrail_dir, i)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (HEIGHT, WIDTH))
        img_rotated = rotate_image(img) # Data augmentation by rotation

        train_images.append([np.array(img).reshape(HEIGHT, WIDTH, 3), [1, 0]])
        train_images.append([np.array(img_rotated), [0, 1]])

    random.shuffle(train_images)

    train_img = np.array([i[0] for i in train_images])
    train_label = np.array([i[1] for i in train_images])
    print("Nb train examples: ", len(train_img))
    return train_img, train_label

def rotate_image(img):
    (h,w,_) = img.shape
    center = (w / 2, h / 2) 
    M = cv2.getRotationMatrix2D(center, 90*random.randint(1,3), 1.0) 
    img2 = cv2.warpAffine(img, M, (h, w)) 
    return img2

def unfreeze(model):
    model.trainable = True
    fine_tune_at = 50
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
    return model


def showStat(acc, val_acc, loss, val_loss, initial_epochs):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs - 1, initial_epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def getCallbackSaver():
  filepath="/content/gdrive/My Drive/DeepRailDrive/ModelSaveJon/epochs:{epoch:03d}-val_acc:{val_acc:.3f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
  callbacks_list = [checkpoint]
  return callbacks_list


train_img, train_label = get_data()
resnet_model = get_model()

init_epoch = 20
fine_tune_epochs = 2
total_epoch = init_epoch + fine_tune_epochs
lr = 1e-3
HEIGHT = 256
WIDTH = 256

resnet_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adam(lr=lr),
                     metrics=['accuracy'])
resnet_model.summary()
history = resnet_model.fit(x=train_img, y=train_label, batch_size=64, epochs=init_epoch, verbose=1, callbacks=getCallbackSaver, validation_split=0.1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# resnet_model = unfreeze(resnet_model)
# resnet_model.compile(loss=keras.losses.categorical_crossentropy,
#                      optimizer=keras.optimizers.Adam(lr=lr/10),
#                      metrics=['accuracy'])
# resnet_model.summary()
# history_fine = resnet_model.fit(x=train_img, y=train_label, batch_size=64, epochs=total_epoch, initial_epoch= init_epoch, verbose=1, callbacks=None,
#                  validation_split=0.1)
# acc += history_fine.history['acc']
# val_acc += history_fine.history['val_acc']

# loss += history_fine.history['loss']
# val_loss += history_fine.history['val_loss']

showStat(acc,val_acc,loss,val_loss, init_epoch)

model_json = resnet_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
resnet_model.save_weights("model.h5")
print("Saved model to disk")