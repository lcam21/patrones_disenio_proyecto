from PIL import Image
from glob import glob
import PIL
import sys
import os


path = '/home/loriaga/ws/maestria/patrones_disenio_proyecto/test_jpg/'
path_originals = '/home/loriaga/ws/maestria/patrones_disenio_proyecto/test/'

if not os.path.exists(path):
    os.makedirs(path)

def processImage():
    listing = os.listdir(path_originals)
    for infile in listing:
        img = Image.open(path_originals+infile)
        name = infile.split('.')
        first_name = path+'/'+name[0] + '.jpg'

        bg = Image.new("RGB", img.size, (0,0,0))
        bg.paste(img,img)
        bg.save(first_name)

processImage()
