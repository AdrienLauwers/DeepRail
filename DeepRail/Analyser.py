import xml.etree.ElementTree as ET
from PIL import Image
import requests
from io import BytesIO
import math
import random


if __name__ == "__main__":
    stride = 0.000314
    mid = stride/2
    lat = 50.7105615
    lng = 4.38
    for i in range(10):
        for j in range(10):
            response = requests.get(
                    "https://maps.googleapis.com/maps/api/staticmap?center=" +
                    str(lat)+","+str(lng)+"&"
                    "zoom=20&"
                    "size=276x276&"
                    "maptype=satellite&"
                    "key=AIzaSyBxyQwsLszcHb5E1tpOHp_wOu2MEaYx4C8"
                    )
            img = Image.open(BytesIO(response.content))
            w, h = img.size
            margin = 20
            img.crop((margin, margin, w - margin, h - margin)).save("dataset/analyse" + str(str(i)+str(j)).zfill(5) + ".png", "png")
            lng += stride
        lat +=stride