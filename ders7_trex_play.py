# trex_play

from keras.models import model_from_json
import os
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss


mon = {'top':360, 'left':150, 'width':250, 'height':100}
sct = mss()

width = 125
height = 50

model = model_from_json(open("model.json", "r").read())
model.load_weights("trex_weights.h5")

print(model.summary())

labels = ["Down", "Right", "Up"]

# değişkenler
framerate_time = time.time()
counter = 0
i = 0
current_framerate = 0
delay = 0.4 # gecikme süresi
count_time = 0
last_result = 1 # son sonuç  

while True:
    time_start = time.time() # Döngünün başlangıç zamanını kaydeder. Bu, kare hızını (frame rate) hesaplamak için kullanılır.
    img = sct.grab(mon) # Ekranın belirli bir bölgesini (mon değişkeni ile belirtilen) yakalar ve bir görüntü nesnesi oluşturur.
    img = Image.frombytes('RGB', img.size, img.rgb) # # Yakalanan görüntüyü RGB formatına dönüştürür.
    img = np.array(img.convert('L').resize((width, height))) # Görüntüyü gri tonlamalı yapar, yeniden boyutlandırır ve bir NumPy dizisine dönüştürür.
    img = img / 255 # # Görüntüdeki piksel değerlerini 0 ile 1 arasına ölçeklendirir.

    time_grab = time.time() -  time_start # Görüntü yakalama süresini hesaplar.
    time_start = time.time() # Zamanı sıfırlar.

    X = np.array([img]) # Görüntüyü bir NumPy dizisine dönüştürür.
    X = X.reshape(X.shape[0], width, height, 1) # Görüntüyü CNN (evrişimsel sinir ağı) modeline uygun hale getirir.
    r = model.predict(X)
    result = np.argmax(r) # Tahmin edilen sınıfı belirler. 

    time_predict = time.time() - time_start # Tahmin süresini hesaplar.

    # 0 > down,  1 > right   2 > up
    if ((result == 0) and (last_result != 0)): # Eğer tahmin sonucu 0 ise VE bir önceki sonuç 0 değilse:
        keyboard.press(keyboard.KEY_DOWN)  # Klavyede 'aşağı' tuşuna bas.
        print("if çalıştı ", result)
    elif result == 2:
        if last_result == 0: # bir önceki adımda Trex eğilmiştir
            keyboard.release(keyboard.KEY_DOWN)
        keyboard.press_and_release(keyboard.KEY_UP)
        print("elif çalıştı ", result)
    last_result = result # Son sonucu günceller.
    counter = 1

    if (time.time() - framerate_time) > 1:  # Eğer geçen süre 1 saniyeden fazlaysa:

        current_framerate = counter /( time.time() - framerate_time)
        counter = 0
        framerate_time = time.time() # şimdiki zamanı sakla bir sonraki turda bir önceki zaman olması
        delay -= 0.0001

        if delay < 0:
            delay = 0

    # os.system("cls")
    # i += 1
    time.sleep(delay)




