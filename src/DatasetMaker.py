import xml.etree.ElementTree as ET
from PIL import Image
import requests
from io import BytesIO
import math
import random
import os
import sys

def get_api_key():
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../api.txt')
    file = open(PATH, "r")
    lines = file.readlines()

    API_keys = {}
    for line in lines:
        l = line.replace('\n', '').split(":")
        API_keys[l[0]] = str(l[1])

        return API_keys


def get_distance(lat_1, lng_1, lat_2, lng_2):
    d_lat = lat_2 - lat_1
    d_lng = lng_2 - lng_1
    temp = (
         math.sin(d_lat / 2) ** 2
       + math.cos(lat_1)
       * math.cos(lat_2)
       * math.sin(d_lng / 2) ** 2
    )
    return 6373.0 * (2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp)))

def get_nodes():
    counter = 0
    selected_nodes = []
    all_nodes = []
    for child in root:
        if child.tag == "node":
            lat = child.attrib['lat']
            long = child.attrib['lon']
            all_nodes.append({"lat": float(lat), "lng": float(long)})
            if counter%STRIDE == 0 and counter > OFFSET*STRIDE :
                selected_nodes.append({"lat": float(lat), "lng": float(long)})
            counter += 1
            if counter > (QUANTITY + OFFSET) * STRIDE :
                break
    print(str(len(all_nodes)) + " positions found.\n"+str(len(selected_nodes))+" positions selected.")
    return all_nodes,selected_nodes

'''
"https://maps.googleapis.com/maps/api/staticmap?center=" +
str(child["lat"]) + "," + str(child["lng"]) +
"zoom=20&"
"size=276x276&"
"maptype=satellite&"
"key=" + API_key
'''

def generate_railimg(nodes):
    counter = 0
    for child in nodes:
        strCoord = str(child["lat"]) + "," + str(child["lng"])
        response = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap?"
            "center=" +strCoord+ "&"
            "zoom="+str(ZOOM)+"&"
            "size="+str(int(GLOBALSIZE))+"x"+str(int(GLOBALSIZE))+"&"
            "maptype=satellite&"
            "key="+str(API_key)
        )
        img = Image.open(BytesIO(response.content))
        w, h = img.size

        #img.crop((0, 0, w - MARGIN, h - MARGIN)).save("dataset/rail/brut" + str(counter).zfill(5) + ".png", "png")
        img.crop((0, 0, w - (CROPSIZE+MARGIN), h - (CROPSIZE+MARGIN))).save("dataset/rail/HG-" + strCoord + ".png", "png")
        img.crop((CROPSIZE, 0,  w - (MARGIN), h - (CROPSIZE+MARGIN))).save("dataset/rail/HD-" + strCoord + ".png", "png")
        img.crop((0, CROPSIZE, w - (CROPSIZE+MARGIN), h - ( MARGIN))).save("dataset/rail/BG-" + strCoord + ".png", "png")
        img.crop((CROPSIZE, CROPSIZE, w - (MARGIN), h - (MARGIN))).save("dataset/rail/BD-" + strCoord + ".png", "png")
        img.crop((CROPSIZE/2, CROPSIZE/2, w - (CROPSIZE/2 + MARGIN), h - (CROPSIZE/2 + MARGIN))).save("dataset/rail/MI-" + strCoord + ".png", "png")
        counter+=1
        print("rail"+strCoord+".png generated")

def generate_nonrailimg(nodes):
    counter = 0
    for i in range(QUANTITY):
        found = False
        lat = 0
        lng = 0
        while not found :
            lat = LAT_SOURCE + ((random.random()-0.5) / 100)
            lng = LNG_SOURCE + ((random.random()-0.5) / 100)
            found = True
            for child in nodes:
                if get_distance(lat, lng, child["lat"], child["lng"]) < GLOBALSIZE:
                    found = False
        strCoord = str(lat) + "," + str(lng)
        response = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap?"
            "center=" + strCoord + "&"
            "zoom=" + str(ZOOM) + "&"
            "size=" + str(int(SIZE)) + "x" + str(int(SIZE)) + "&"
            "maptype=satellite&"
            "key=" + str(API_key)
        )
        img = Image.open(BytesIO(response.content))
        w, h = img.size
        img.crop((0, 0, w - MARGIN, h - MARGIN)).save("dataset/nonrail/" + strCoord + ".png", "png")
        counter+=1
        print("nonrail"+strCoord+".png generated")

#Global param
SIZE = 256
ZOOM = 21
MARGIN = 20
QUANTITY = 10 #float('inf')

#rail
STRIDE = 2
OFFSET = 0

#non rail
DISTANCE_RAIL = 0.05
LAT_SOURCE = 50.409
LNG_SOURCE = 4.4404

CROPSIZE = SIZE/2
GLOBALSIZE = SIZE+CROPSIZE+MARGIN

API_keys = get_api_key()
API_key = API_keys["GoogleMaps"]
tree = ET.parse('50,4,51,5.xml')
root = tree.getroot()
all_nodes, selected_nodes = get_nodes()
#generate_railimg(selected_nodes, GLOBALSIZE, CROPSIZE)
generate_nonrailimg(all_nodes)

