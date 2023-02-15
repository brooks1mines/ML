import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from PIL import Image

folder = ''
label = ''
file_path = []
images = []
def find_images(movement):
    if movement == 'Baroque':
        file_path = []
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Baroque'
        files = os.listdir(folder)
        for i in files:
            file_path.append(os.path.join(folder, i))
        return file_path
        
    elif movement == 'Cubism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Cubism'
        files = os.listdir(folder)
        return files
    elif movement == 'Expressionism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Expressionism'
        files = os.listdir(folder)
        return files
    elif movement == 'Fauvism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Fauvism'
        files = os.listdir(folder)
        return files
    elif movement == 'Impressionism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Impressionism'
        files = os.listdir(folder)
        return files
    elif movement == 'NeoClassicism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/NeoClassicism'
        files = os.listdir(folder)
        return files
    elif movement == 'PostImpressionism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/PostImpressionism'
        files = os.listdir(folder)
        return files
    elif movement == 'Realism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Realism'
        files = os.listdir(folder)
        return files
    elif movement == 'Renaissance':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Renaissance'
        files = os.listdir(folder)
        return files
    elif movement == 'Rococo':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Rococo'
        files = os.listdir(folder)
        return files
    elif movement == 'Romanticism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Romanticism'
        files = os.listdir(folder)
        return files
    elif movement == 'Surrealism':
        folder = 'C:/Users/Jeremy/Desktop/art/Movements/Historical Movements/Surrealism'
        files = os.listdir(folder)
        return files
    else:
        print("Art movement not available, check spelling")

def turn_images_to_tensors(file_path):
    for i in file_path:
        image = tf.io.read_file(i)
        image = tf.image.decode_image(image, channels = 3)
        image = np.array(image)
        images.append(image)
    images = tf.convert_to_tensor(images, dtype=tf.float32)

stuff = find_images('Baroque')
print(turn_images_to_tensors(stuff))