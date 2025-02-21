import keyboard
import os
import uuid
from mss import mss
from PIL import Image
import time 
import sys

mon = {'top':360, 'left':150, 'width':250, 'height':100}
sct = mss()

i = 0

def record_screen(record_id, key):
    global i
    i += 1
    print(f"{key}:{i}")
    img = sct.grab(mon)
    im = Image.frombytes('RGB', img.size, img.rgb)
    im.save(f"img/{key}_{record_id}_{i}.png")

record_id = uuid.uuid4()

# global kavramı
# döngü ile resim kaydet
# gemini ile Yapay sinir ağı simülasyon adımlarını öğreneceğiz 

is_exit = False # Boolean veri tipi 1-0, T/F 

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey('esc', exit)

while True:
    if is_exit:
        break
    
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1)
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError:
        continue
