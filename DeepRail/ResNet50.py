import keras
import numpy as np
import os
from keras.applications import resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import cv2
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

def get_model():
  
    input_tensor = keras.Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

    # create the base pre-trained model
    base_model = resnet50.ResNet50(input_tensor=input_tensor,weights='imagenet',include_top=False)

    for layer in base_model.layers:
        layer.trainable=False

    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)
    x = keras.layers.Dense(2, activation='softmax')(x)

    updatedModel = keras.Model(base_model.input, x)

    return updatedModel

if __name__ == "__main__":
    resnet_model = get_model()

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

    train_dir = os.path.join(PATH, 'all')
    train_rail_dir = os.path.join(train_dir, 'rail') 
    train_nonrail_dir = os.path.join(train_dir, 'nonrail')

    train_rail_size = len(os.listdir(train_rail_dir))
    train_nonrail_size = len(os.listdir(train_nonrail_dir))

    # validation_dir = os.path.join(PATH, 'validation')
    # validation_rail_dir = os.path.join(train_dir, 'rail') 
    # validation_nonrail_dir = os.path.join(train_dir, 'nonrail') 

    # validation_rail_size = len(os.listdir(validation_rail_dir))
    # validation_nonrail_size = len(os.listdir(validation_nonrail_dir))

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

    random.shuffle(train_images)

    train_img = np.array([i[0] for i in train_images]).reshape(-1,224,224,3)
    train_label = np.array([i[1] for i in train_images])

    print(len(train_images))

    # filename = os.path.join(train_rail_dir, 'rail00000.png')

    # load an image in PIL format
    # original_image = load_img(filename, target_size=(224, 224))
    # convert the PIL image (width, height) to a NumPy array (height, width, channel)
    # numpy_image = img_to_array(original_image)
    # Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.
    # input_image = np.expand_dims(numpy_image, axis=0)

    # print('PIL image size = ', original_image.size)
    # print('NumPy image size = ', numpy_image.shape)
    # print('Input image size = ', input_image.shape)
    # plt.imshow(np.uint8(input_image[0]))
    # plt.show()

    # processed_image_resnet50 = resnet50.preprocess_input(input_image.copy())

    # predictions_resnet50 = resnet_model.predict(processed_image_resnet50)
    # label_resnet50 = decode_predictions(predictions_resnet50)
    # print('label_resnet50 = ', label_resnet50)

    resnet_model.compile(loss=keras.losses.categorical_crossentropy,    
                  optimizer=keras.optimizers.Adam(lr=1e-3),
                  metrics=['accuracy'])

    resnet_model.fit(x=train_img, y=train_label, batch_size=64, epochs=3,
                    verbose=1, callbacks=None, validation_split=0.1)

    resnet_model.summary()

    # serialize model to JSON
    model_json = resnet_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    resnet_model.save_weights("model.h5")
    print("Saved model to disk")
