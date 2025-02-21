import keyboard
import os
import uuid
from mss import mss
from PIL import Image
import time 
import sys

mon = {'top':360, 'left':150, 'width':250, 'height':100}
mon2 = {'top':400, 'left':250, 'width':50, 'height':50}
sct = mss()

i = 0

def record_screen(record_id, key):
    global i
    i += 1
    print(f"{key}:{i}")
    img = sct.grab(mon)
    im = Image.frombytes('RGB', img.size, img.rgb)
    im.save(f"img2/{key}_{record_id}_{i}.png")

record_id = uuid.uuid4()

time.sleep(10)
record_screen(record_id, "up")


