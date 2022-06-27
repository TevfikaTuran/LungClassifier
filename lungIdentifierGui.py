import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont

from matplotlib.pyplot import new_figure_manager
from metods import *
import string
import time
import hashlib
import os
import requests
import traceback
from PIL import Image
from threading import Thread
from matplotlib import image


def selectFileDialog():
    file_path = filedialog.askopenfilename()
    return file_path

def maskImage(file_path):
    new_file_name = "img.png"
    
    x = np.zeros((1, 128, 128))

    img = readImages(file_path,  IMG_HEIGHT = 128, IMG_WIDTH = 128, is_gray=0)
    x[0] = img
    
    model = load_model("model.h5")
    pred_img = model.predict(x/255.)*255.0
    
    img = fillHoles(pred_img[0])
    img = removeSmallObjects(img, min_area = 200)
    
    image.imsave("cache\\" + new_file_name, img, cmap=plt.cm.gray)
    
    img = readImages("cache\\" + new_file_name, 500, 500, 0)
    image.imsave("cache\\" + "resized_" + new_file_name, img, cmap=plt.cm.bone)
    return ("cache\\" + new_file_name), ("cache\\" + "resized_" + new_file_name)

def segmentImage(lung_path, mask_path):
    save_file_name = "img2.png"
    mask = readImages(mask_path,  IMG_HEIGHT = 128, IMG_WIDTH = 128, is_gray=0)
    _ , mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY) #binary img
    img = readImages(lung_path,  IMG_HEIGHT = 128, IMG_WIDTH = 128, is_gray=0)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    image.imsave("cache\\" + save_file_name, segmented_img, cmap=plt.cm.bone)
    img = readImages("cache\\" + save_file_name, 500, 500, 0)
    image.imsave("cache\\" + "resized_" + save_file_name, img, cmap=plt.cm.bone)
    return ("cache\\" + save_file_name), ("cache\\" + "resized_" + save_file_name)

def predictImage(segment_path):
    model = load_model("models\\" + "LungIdentifierModelVGG16Testing.h5")
    
    img = readImages(segment_path, 128, 128, 0)
    
    x = np.zeros((1, 128, 128))
    
    x[0] = img
    
    x = np.repeat(x[..., np.newaxis], 3, -1)
    x = x.reshape(-1, 128, 128, 3)

    preds = np.argmax(model.predict(x/255.0), axis=-1)
    
    return preds[0]

class App:
    def __init__(self, root):
        # setting title
        root.title("Akciğer Segmentasyonu ve Pnömotoraks Tespiti")
        # setting window size
        width = 592
        height = 661
        
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height,
                                    (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        GLabel_891 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        GLabel_891["font"] = ft
        GLabel_891["fg"] = "#333333"
        GLabel_891["justify"] = "center"
        GLabel_891["text"] = ""
        GLabel_891.place(x=0, y=0, width=591, height=622)
        self.label = GLabel_891
        
        GLabel_892 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        GLabel_892["font"] = ft
        GLabel_892["fg"] = "#333333"
        GLabel_892["justify"] = "center"
        GLabel_892["text"] = "Yüklenen görsel tipi: Görsel bekleniyor..."
        GLabel_892.place(x=150, y=40, width=300, height=20)
        self.label2 = GLabel_892
        
        GLabel_893 = tk.Label(root)
        ft = tkFont.Font(family='Times', size=10)
        GLabel_893["font"] = ft
        GLabel_893["fg"] = "#333333"
        GLabel_893["justify"] = "center"
        GLabel_893["text"] = "Hastalık tipi: Görsel bekleniyor..."
        GLabel_893.place(x=150, y=570, width=300, height=20)
        self.label3 = GLabel_893

        GButton_665 = tk.Button(root)
        GButton_665["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_665["font"] = ft
        GButton_665["fg"] = "#000000"
        GButton_665["justify"] = "center"
        GButton_665["text"] = "Dosya Seç"
        GButton_665.place(x=260, y=630, width=70, height=25)
        GButton_665["command"] = self.GButton_665_command

        GButton_207 = tk.Button(root)
        GButton_207["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_207["font"] = ft
        GButton_207["fg"] = "#000000"
        GButton_207["justify"] = "center"
        GButton_207["text"] = "Geri"
        GButton_207.place(x=100, y=600, width=70, height=25)
        GButton_207["command"] = self.GButton_207_command

        GButton_162 = tk.Button(root)
        GButton_162["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_162["font"] = ft
        GButton_162["fg"] = "#000000"
        GButton_162["justify"] = "center"
        GButton_162["text"] = "İleri"
        GButton_162.place(x=420, y=600, width=70, height=25)
        GButton_162["command"] = self.GButton_162_command


    #Dosya seçim işlemleri
    def GButton_665_command(self):
        file_path = selectFileDialog()
        img = readImages(file_path, 500, 500, 0)
        image.imsave("cache\\" + "orig.png", img, cmap=plt.cm.gray)
        
        self.original_img = "cache\\" + "orig.png"
        
        self.selection = "segment"
        masked_image_path, resized_masked = maskImage(file_path)
        self.mask = resized_masked

        segmented_image_path, resized_segmented = segmentImage(file_path, masked_image_path)
        segmented_image = Image.open(segmented_image_path)
        self.segment = resized_segmented
        
        predicted_type = predictImage(segmented_image_path)
        if predicted_type == 0: predicted_type = "Sağlıklı" 
        else: predicted_type = "Pnömotoraks"
        self.label3['text'] = f"Tahmin Edilen Tanı: {predicted_type}"
        
        self.label2['text'] = f"Yüklenen görsel tipi: Segment"

        self.img = tk.PhotoImage(file=resized_segmented)
        self.label.configure(image=self.img)
        self.label.image = self.img

    def GButton_207_command(self):
        if self.selection == "mask":
            self.selection = "orig"
            self.img = tk.PhotoImage(file=self.original_img)
            self.label.configure(image=self.img)
            self.label.image = self.img
            self.label2['text'] = "Yüklenen görsel tipi: Orijinal"
            return
        if self.selection == "segment":
            self.selection = "mask"
            self.img = tk.PhotoImage(file=self.mask)
            self.label.configure(image=self.img)
            self.label.image = self.img
            self.label2['text'] = "Yüklenen görsel tipi: Maske"
            return
        if self.selection == "orig":
            self.selection = "segment"
            self.img = tk.PhotoImage(file=self.segment)
            self.label.configure(image=self.img)
            self.label.image = self.img
            self.label2['text'] = "Yüklenen görsel tipi: Segment"
            return

    def GButton_162_command(self):
        if self.selection == "mask":
            self.selection = "segment"
            self.img = tk.PhotoImage(file=self.segment)
            self.label.configure(image=self.img)
            self.label.image = self.img
            self.label2['text'] = "Yüklenen görsel tipi: Segment"
            return
        if self.selection == "segment":
            self.selection = "orig"
            self.img = tk.PhotoImage(file=self.original_img)
            self.label.configure(image=self.img)
            self.label.image = self.img
            self.label2['text'] = "Yüklenen görsel tipi: Orijinal"
            return
        if self.selection == "orig":
            self.selection = "mask"
            self.img = tk.PhotoImage(file=self.mask)
            self.label.configure(image=self.img)
            self.label.image = self.img
            self.label2['text'] = "Yüklenen görsel tipi: Maske"
            return

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
