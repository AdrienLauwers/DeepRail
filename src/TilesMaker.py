import os
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__":

    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    images_dir = os.path.join(PATH, 'images')

    height = 256
    width = 256

    for i in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, i)
        
        if os.path.isdir(img_path): 
            continue

        img = Image.open(img_path, mode="r")

        orig_width, orig_height = img.size
        
        top = 0
        bottom = height

        cnt = 0
        while bottom <= orig_height:
            left = 0
            right = width
            while right <= orig_width:
                im = img.crop((left, top, right, bottom))
                im.save("dataset/images/tiles/" + str(i).replace(".png","") + "_" + str(cnt) + ".png", "PNG")
                left = right + 1
                right += width
                cnt += 1

            top = bottom + 1
            bottom += height        

        



