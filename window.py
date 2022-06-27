from tkinter import *


def btn_clicked():
    print("Button Clicked")


window = Tk()

window.geometry("514x600")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 600,
    width = 514,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = -402, y = 233,
    width = 50,
    height = 50)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b1.place(
    x = -187, y = 233,
    width = 50,
    height = 50)

img2 = PhotoImage(file = f"img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b2.place(
    x = 48, y = 233,
    width = 50,
    height = 50)

canvas.create_text(
    -162.0, 302.0,
    text = "Test",
    fill = "#000000",
    font = ("RobotoRoman-Medium", int(16.0)))

window.resizable(False, False)
window.mainloop()
