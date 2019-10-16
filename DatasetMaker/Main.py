import xml.etree.ElementTree as ET
from PIL import Image
import requests
from io import BytesIO
import math
import random

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

def get_nodes(stride,quantity,decallage):
    counter = 0
    selected_nodes = []
    all_nodes = []
    for child in root:
        if child.tag == "node":
            lat = child.attrib['lat']
            long = child.attrib['lon']
            all_nodes.append({"lat": float(lat), "lng": float(long)})
            if counter%stride == 0 and counter > decallage*stride :
                selected_nodes.append({"lat": float(lat), "lng": float(long)})
            counter += 1
            if counter > (quantity + decallage) * stride :
                break
    print(str(len(all_nodes)) + " positions found.\n"+str(len(selected_nodes))+" positions selected.")
    return all_nodes,selected_nodes

'''
"https://maps.googleapis.com/maps/api/staticmap?center=" +
str(child["lat"]) + "," + str(child["lng"]) +
"zoom=20&"
"size=276x276&"
"maptype=satellite&"
"key=AIzaSyBxyQwsLszcHb5E1tpOHp_wOu2MEaYx4C8"
'''

def generate_railimg(nodes):
    counter = 0
    for child in nodes:
        response = requests.get(
            "https://maps.googleapis.com/maps/api/staticmap?"
            "center=" +str(child["lat"]) + "," + str(child["lng"]) + "&"
            "zoom=20&"
            "size=276x276&"
            "maptype=satellite&"
            "key=AIzaSyBxyQwsLszcHb5E1tpOHp_wOu2MEaYx4C8"
        )
        img = Image.open(BytesIO(response.content))
        w, h = img.size
        margin = 20
        img.crop((margin, margin, w - margin, h - margin)).save("dataset/rail"+str(counter).zfill(5)+".png", "png")
        counter+=1
        print("rail"+str(counter).zfill(5)+".png generated")

def generate_nonrailimg(nodes,quantity, lat_1, lng_1, lat_2, lng_2, rng):
    counter = 0

    for i in range(quantity):
        found = False
        lat = 0
        lng = 0
        while not found :
            lat = lat_1 + random.random() #50.5733647 #
            lng = lng_1 + random.random()  #4.6885536 #
            found = True
            for child in nodes:
                if get_distance(lat, lng, child["lat"], child["lng"]) < rng:
                    found = False
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
        margin = 25
        img.crop((margin, margin, w - margin, h - margin)).save("dataset/nonrail" + str(counter).zfill(5) + ".png", "png")
        counter+=1
        print("nonrail"+str(counter).zfill(5)+".png generated")



tree = ET.parse('50,4,51,5.xml')
root = tree.getroot()
all_nodes, selected_nodes = get_nodes(5,500,1000)
generate_railimg(selected_nodes)
#generate_nonrailimg(all_nodes,1000,50,4,51,5,0.05)

