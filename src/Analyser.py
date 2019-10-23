import xml.etree.ElementTree as ET
from PIL import Image
import requests
from io import BytesIO
import math
import random
import os

def get_api_key():
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../api.txt')
    file = open(PATH, "r")
    lines = file.readlines()

    API_keys = {}
    for line in lines:
        l = line.replace('\n', '').split(":")
        API_keys[l[0]] = l[1]

    return API_keys

if __name__ == "__main__":

    API_keys = get_api_key()
    API_key = API_keys["GoogleMaps"]

    stride = 0.000314
    mid = stride/2
    lat = 50.6906239
    lng_ori = 4.5731336

    for i in range(10):
        lng = lng_ori
        for j in range(10):
            response = requests.get(
                    "https://maps.googleapis.com/maps/api/staticmap?center=" +
                    str(lat)+","+str(lng)+"&"
                    "zoom=20&"
                    "size=276x276&"
                    "maptype=satellite&"
                    "key="+API_key
                    )
            img = Image.open(BytesIO(response.content))
            w, h = img.size
            margin = 20
            img.crop((margin, margin, w - margin, h - margin)).save("../../DeepRailDataset/analyse/img4/limal" + str(str(i)+str(j)).zfill(5) + ".png", "png")
            print(lat,lng)
            lng += stride
        lat +=stride