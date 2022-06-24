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
    
    img = readImages("cache\\" + "resized_" + new_file_name, 300, 300, 0)
    image.imsave("cache\\" + "resized_" + new_file_name, img, cmap=plt.cm.bone)
    return ("cache\\" + new_file_name), ("cache\\" + "resized_" + new_file_name)

def segmentImage(lung_path, mask_path):
    save_file_name = "img2.png"
    mask = readImages(mask_path,  IMG_HEIGHT = 128, IMG_WIDTH = 128, is_gray=0)
    _ , mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY) #binary img
    img = readImages(lung_path,  IMG_HEIGHT = 128, IMG_WIDTH = 128, is_gray=0)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    image.imsave("cache\\" + save_file_name, segmented_img, cmap=plt.cm.bone)
    img = readImages("cache\\" + "resized_" + save_file_name, 300, 300, 0)
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
        GLabel_891["text"] = "label"
        GLabel_891.place(x=0, y=0, width=591, height=622)
        self.label = GLabel_891

        GButton_211 = tk.Button(root)
        GButton_211["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_211["font"] = ft
        GButton_211["fg"] = "#000000"
        GButton_211["justify"] = "center"
        GButton_211["text"] = "Next"
        GButton_211.place(x=510, y=630, width=70, height=25)
        GButton_211["command"] = self.GButton_211_command

        GButton_665 = tk.Button(root)
        GButton_665["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_665["font"] = ft
        GButton_665["fg"] = "#000000"
        GButton_665["justify"] = "center"
        GButton_665["text"] = "Dosya Seç"
        GButton_665.place(x=10, y=630, width=70, height=25)
        GButton_665["command"] = self.GButton_665_command

        GButton_207 = tk.Button(root)
        GButton_207["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_207["font"] = ft
        GButton_207["fg"] = "#000000"
        GButton_207["justify"] = "center"
        GButton_207["text"] = "Download"
        GButton_207.place(x=100, y=630, width=70, height=25)
        GButton_207["command"] = self.GButton_207_command

        GButton_162 = tk.Button(root)
        GButton_162["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=10)
        GButton_162["font"] = ft
        GButton_162["fg"] = "#000000"
        GButton_162["justify"] = "center"
        GButton_162["text"] = "Deny"
        GButton_162.place(x=420, y=630, width=70, height=25)
        GButton_162["command"] = self.GButton_162_command

        GLineEdit_685 = tk.Entry(root)
        GLineEdit_685["borderwidth"] = "1px"
        ft = tkFont.Font(family='Times', size=10)
        GLineEdit_685["font"] = ft
        GLineEdit_685["fg"] = "#333333"
        GLineEdit_685["justify"] = "center"
        GLineEdit_685["text"] = ""
        GLineEdit_685.place(x=210, y=630, width=170, height=25)
        self.textbox = GLineEdit_685

    def GButton_211_command(self):
        code = next_code(self.textbox.get().replace('https://prnt.sc/', ''))
        self.textbox.delete("0", tk.END)
        self.textbox.insert("0", "https://prnt.sc/" + code)
        print(code)

        url = get_img_url(code)

        img_path = get_img(url)

        self.img = tk.PhotoImage(file=img_path)
        self.label.configure(image=self.img)
        self.label.image = self.img

    #Dosya seçim işlemleri
    def GButton_665_command(self):
        file_path = selectFileDialog()
        
        masked_image_path, resized_masked = maskImage(file_path)
        masked_image = Image.open(masked_image_path)
        
        segmented_image_path, resized_segmented = segmentImage(file_path, masked_image_path)
        segmented_image = Image.open(segmented_image_path)

        predicted_type = predictImage(segmented_image_path)
        
        self.textbox.delete("0", tk.END)
        self.textbox.insert("0", f"Tahmin edilen tip: {predicted_type}")

        self.img = tk.PhotoImage(file=resized_segmented)
        self.label.configure(image=self.img)
        self.label.image = self.img

    def GButton_207_command(self):
        print("command")

    def GButton_162_command(self):
        print("command")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
