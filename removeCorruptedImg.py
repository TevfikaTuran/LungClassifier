

import os
from metods import readImages

def fixImages(folder_name):
    for img_name in os.listdir(folder_name):
        try:
            readImages(folder_name + img_name)
        except Exception as ex:
            os.remove(folder_name + img_name)
            
path = "C:\\Users\\ugur_\\Python Projects\\LungClassifier\\Healty\\"


fixImages(path)