import keras
import numpy as np
import os
from tqdm import tqdm
import cv2
import random

# load json and create model
json_file = open('resnet50_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
resnet50_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
resnet50_model.load_weights("resnet50_model.h5")
print("Loaded model from disk")
 
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

test_dir = os.path.join(PATH, 'test')
test_rail_dir = os.path.join(test_dir, 'rail') 
test_nonrail_dir = os.path.join(test_dir, 'nonrail')

test_images = []
test_path = []

for i in tqdm(os.listdir(test_rail_dir)):
        img_path = os.path.join(test_rail_dir, "rail00061b.png")
        test_path.append(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        test_images.append([np.array(img), [1, 0]])

for i in tqdm(os.listdir(test_nonrail_dir)):
    img_path = os.path.join(test_nonrail_dir, i)
    test_path.append(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
print(predicts)
predicts = np.around(predicts)
print(predicts)
print(test_label[0])
for i in range(len(predicts)):
    if int(predicts[i*2]) != test_label[i][0]:
        print (i, predicts[i], test_label[i], test_path[i])