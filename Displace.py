from PIL import Image
from glob import glob
import PIL
import sys
import os
import numpy as np
import math
import skimage
import random

path = 'C:/Users/ajskr/Documents/Maestria/Cursos/Sistemas Empotrados de Alto Desempeno/Proyecto3/test2/'
path_originals = 'C:/Users/ajskr/Documents/Maestria/Cursos/Sistemas Empotrados de Alto Desempeno/Proyecto3/Pruebas_jpg/'

if not os.path.exists(path):
    os.makedirs(path)

def displaceImage():
    listing = os.listdir(path_originals)
    for infile in listing:
        img = Image.open(path_originals+infile)
        name = infile.split('.')
        #first_name = path+'/'+'R_'+name[0] + '.jpg'
        first_name = path+'/'+indice+name[0] + '.jpg'
        img = img.crop((-64,-64,192,192))
        desplx = random.randint(0,40)
        desply = random.randint(0,40)
        delta = 80 + random.randint(30,50)
        xi = 40 + desplx
        yi = 40 + desply
        xf = xi + delta
        yf = yi + delta
        img = img.crop((xi,yi,xf,yf))
        img = img.resize((128,128), Image.ANTIALIAS)
        img.save(first_name)


indice = 'R1_'
displaceImage()
indice = 'R2_'
displaceImage()