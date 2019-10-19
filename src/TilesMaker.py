import os
from tqdm import tqdm
from PIL import Image


def crop_orig(pixel_crop, images_dir):
    print("Croping original files")
    for i in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, i)
        if os.path.isdir(img_path): 
            continue
        img = Image.open(img_path, "r")
        w, h = img.size
        im = img.crop((pixel_crop, pixel_crop, w - pixel_crop, h - pixel_crop))
        im.save("../DeepRailDataset/DOTA/part3/cropped/" + i, "PNG")

    return

def make_tiles(width, height, images_dir):
    print("Creating tiles")
    for i in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, i)
        img = Image.open(img_path, "r")
        if os.path.isdir(img_path): 
            continue

        orig_width, orig_height = img.size
        
        top = 0
        bottom = height

        cnt = 0
        while bottom <= orig_height:
            left = 0
            right = width
            while right <= orig_width:
                im = img.crop((left, top, right, bottom))
                im.save("tiles/" + str(i).replace(".png","") + "_" + str(cnt) + ".png", "PNG")
                left = right + 1
                right += (width + 1)
                cnt += 1

            top = bottom + 1
            bottom += (height + 1)
    return

if __name__ == "__main__":

    directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    tiles = os.path.join(directory, 'DeepRailDataset/DOTA')
    images_dir = os.path.join(tiles, 'part3')

    height = 256
    width = 256

    pixel_crop = 100

    # crop_orig(pixel_crop, images_dir)

    images_dir = os.path.join(images_dir, 'cropped')
    make_tiles(width, height, images_dir)