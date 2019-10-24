import keras
from keras import Input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.applications import resnet50
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

WIDTH = 258
HEIGHT = 258

# load json and create model
json_file = open('../models/resnet50_epoch55.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
resnet50_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
resnet50_model.load_weights("../models/resnet50_epoch55.h5")
print("Loaded model from disk")
 
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#test_dir = os.path.join(os.path.join(PATH, '../../DeepRailDataset'), 'test')
#test_rail_dir = os.path.join(test_dir, 'rail')
#test_nonrail_dir = os.path.join(test_dir, 'nonrail')

test_analyse = os.path.join(os.path.join(os.path.join(PATH, 'DeepRailDataset'), 'analyse'), 'img7')

test_images = []
test_path = []

# for i in tqdm(os.listdir(test_rail_dir)):
# 	img_path = os.path.join(test_rail_dir, i)
# 	test_path.append(img_path)
# 	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# 	img = cv2.resize(img, (224, 224))
# 	test_images.append([np.array(img), [1, 0]])

# for i in tqdm(os.listdir(test_nonrail_dir)):
#     img_path = os.path.join(test_nonrail_dir, i)
#     test_path.append(img_path)
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (224, 224))
#     test_images.append([np.array(img), [0, 1]])

for i in tqdm(sorted(os.listdir(test_analyse))):
	img_path = os.path.join(test_analyse, i)
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	if img is None:
		continue
	test_path.append(img_path)
	img = cv2.resize(img, (224, 224))
	test_images.append([np.array(img), [0, 1]])

# random.shuffle(test_images)

test_img = np.array([i[0] for i in test_images]).reshape(-1,224,224,3)
test_label = np.array([i[1] for i in test_images])


# evaluate loaded model on test data

resnet50_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
# score = resnet50_model.evaluate(test_img, test_label, verbose=1)
# print("%s: %.2f%%" % (resnet50_model.metrics_names[1], score[1]*100))

predicts = resnet50_model.predict(test_img, verbose=1).reshape((-1,))

# equals = np.equal(np.around(predicts) != test_label)
predicts_rounded = np.around(predicts)
for i in range(int(len(predicts)//2)):
	if (predicts[i*2] > predicts[i*2+1]):
		print(test_path[i], "rail", predicts[i*2])
	else:
		print(test_path[i], "non-rail", predicts[i*2+1])
    # if int(predicts[i*2]) != test_label[i][0]:
    #     print (i, predicts[i], test_label[i], test_path[i])

full_img = Image.new('RGB', (WIDTH * 10, HEIGHT * 10))
tiles = map(cv2.imread, test_path)

cnt = 0
x_offset = 0
y_offset = HEIGHT * 9

print(len(test_path))
for img in tiles:
  print(test_path[cnt])
  if (predicts[cnt*2] > predicts[cnt*2+1]):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (256, 256), (0, 0, 200), -1)
    alpha = 0.15
    image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    texted_image =cv2.putText(img=np.copy(image_new), text=str("%.2f" % predicts[cnt*2]), org=(10,40),fontFace=2, fontScale=1.5, color=(255,0,0), thickness=2)
    full_img.paste( Image.fromarray(texted_image), (x_offset, y_offset))
  else:
    full_img.paste( Image.fromarray(img), (x_offset, y_offset))
  x_offset += WIDTH

  cnt += 1
  if (cnt % 10 == 0):
    x_offset = 0
    y_offset -= HEIGHT

full_img.save("../../DeepRailDataset/analyse/img7/result/result.png", "PNG")
