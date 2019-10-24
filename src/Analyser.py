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

def get_next_coord(coord, dist):
    K = 1852*60 # 1 mille marin  = 1852 m; à multiplier par 60 pour passer aux degrés.
    delta = dist/K
    return coord + delta # * cos(0) = 1

def meters_per_pixel_NS(lat, zoom): 
    return 156543.03392 * math.cos(math.radians(lat)) / 2**zoom

def meters_per_pixel_EW(zoom):
    return 156543.03392 / 2**zoom


# if __name__ == "__main__":
#     # lat = 50.0
#     # lng = 90.0
#     lat = 50.677627
#     lng = 4.562762
#     dist_NS = 256 * meters_per_pixel_NS(lat, 20)
#     dist_EW = 256 * meters_per_pixel_EW(20)
#     # print(dist)
#     new_lat = get_next_lat(lat, dist_NS)
#     new_long = get_next_long(lng, lat, dist_EW)
#     print("-------------")
#     print("new lat", new_lat, (new_lat-lat))
#     print("new long", new_long, (new_long-lng))


if __name__ == "__main__":

    API_keys = get_api_key()
    API_key = API_keys["GoogleMaps"]

    # stride = 0.000314
    # mid = stride/2
    lat = 50.7605474
    lng_ori = 4.4586163

    dist_NS = 256 * meters_per_pixel_NS(lat, 20)
    dist_EW = 256 * meters_per_pixel_EW(20)

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
            img.crop((0, 0, w - margin, h - margin)).save("../../DeepRailDataset/analyse/img7/test" + str(str(i)+str(j)).zfill(5) + ".png", "png")
            print(lat,lng)
            lng = get_next_coord(lng, dist_EW)
            # lng += stride
        # lat +=stride
        lat = get_next_coord(lat, dist_NS)