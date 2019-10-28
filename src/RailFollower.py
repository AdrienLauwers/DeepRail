import keras
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import resnet50
from keras.callbacks import *
from keras import Input
from keras.models import load_model

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import cv2
import requests
from io import BytesIO
import math
import random

def load_model_local(filepath):
    # load json and create model
    json_file = open("/content/gdrive/My Drive/DeepRailDrive/model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    resnet50_loaded = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    resnet50_loaded.load_weights('/content/gdrive/My Drive/DeepRailDrive/ModelSave/' + filepath)
    print("Loaded model from disk")

    '''
    resnet50_loaded  = keras.models.load_model('/content/gdrive/My Drive/DeepRailDrive/ModelSave/epochs:050-val_acc:0.993.h5')
     '''
    return resnet50_loaded

lat_o = 50.591429
lng_o =5.443701
resnet_model = load_model_local("epochs:055-val_acc:0.999.h5")


def predict(lat, lng):
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/staticmap?center=" +
        str(lat) + "," + str(lng) + "&"
                                    "zoom=20&"
                                    "size=276x276&"
                                    "maptype=satellite&"
                                    "key=AIzaSyBVaAvXbSQgbOVgjwLUBuZIwKjJdWq6jek"
        , stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img_ori = img[0:256, 0:256]
    img = cv2.resize(np.array(img_ori), (224, 224))
    test_images = [([np.array(img), [0, 1]])]
    test_img = np.array([i[0] for i in test_images]).reshape(-1, 224, 224, 3)
    test_label = np.array([i[1] for i in test_images])
    predicts = resnet_model.predict(test_img, verbose=1).reshape((-1,))
    return predicts, img_ori


def rail(pred):
    if pred[0] > 0.4:
        return True
    else:
        return False


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def get_next_coord(coord, dist, s=1):
    K = 1852 * 60  # 1 nautical marin  = 1852 m; to multiply by 60 to get degrees
    delta = (dist / K) * s
    return coord + delta


def meters_per_pixel_NS(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** zoom


def meters_per_pixel_EW(zoom):
    return 156543.03392 / 2 ** zoom


dist_NS = 256 * meters_per_pixel_NS(lat_o, 20)
dist_EW = 256 * meters_per_pixel_EW(20)

# full_img = Image.new('RGB', (5120,5120))
L = [(lat_o, lng_o, None)]
res = []
counter = 0
while L:
    lat, lng, d = L.pop()
    p, img = predict(lat, lng)

    ''' 
    print(d)
    print(p)
    imgplot = plt.imshow(img)
    plt.show()
    '''
    if rail(p) is True:
        print("FOUND")

        counter += 1
        if counter > 100 :
            break

        res.append((lat, lng))

        lat_N = get_next_coord(lat, dist_NS, 1)
        lat_S = get_next_coord(lat, dist_NS, -1)
        lng_E = get_next_coord(lng, dist_EW, 1)
        lng_W = get_next_coord(lng, dist_EW, -1)

        down = (lat_S, lng, "DOWN")
        up = (lat_N, lng, "UP")
        left = (lat, lng_W, "LEFT")
        right = (lat, lng_E, "RIGHT")
        up_left = (lat_N, lng_W, "UP_LEFT")
        up_right = (lat_N, lng_E, "UP_RIGHT")
        down_left = (lat_S, lng_W, "DOWN_LEFT")
        down_right = (lat_S, lng_E, "DOWN_RIGHT")

        if d is None:

            L.append(down)
            L.append(up)
            L.append(left)
            L.append(right)
            L.append(up_left)
            L.append(up_right)
            L.append(down_left)
            L.append(down_right)
        elif d is "UP":
            L.append(up_left)
            L.append(up_right)
            L.append(up)
        elif d is "DOWN":
            L.append(down_left)
            L.append(down_right)
            L.append(down)
        elif d is "LEFT":
            L.append(up_left)
            L.append(down_left)
            L.append(left)
        elif d is "RIGHT":
            L.append(down_right)
            L.append(right)
            L.append(up_right)
        elif d is "UP_LEFT":
            L.append(up_left)
            L.append(up)
            L.append(left)
        elif d is 'DOWN_LEFT':
            L.append(down_left)
            L.append(left)
            L.append(down)
        elif d is "UP_RIGHT":
            L.append(up_right)
            L.append(up)
            L.append(right)
        elif d is "DOWN_RIGHT":
            L.append(down_right)
            L.append(down)
            L.append(right)

res.sort(key=lambda tup: tup[1])
for r in res:
  print(str(r[0])+"\t"+str(r[1]))