# Öğreneceklerimiz

# 1. requiremens.txt kullanımı ve modüllerin yüklenmesi
# 2. Label encoder - One hot encoder
# 3. Resim isimlerini işleme ve split fonksiyonu
# 4. veri setini bölme -> test ve eğitim veri seti

import glob # Dosyaları kolayca bulmak için kullanılı
import os # Dosya işlemleri yapmak için gerekli.
import random 
import numpy as np # Matematiksel işlemler ve diziler için çok kullanışlı.
import keras # Yapay zeka modelleri kurmamıza yarayan popüler bir kütüphane.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image # # Görüntüleri işlemek için kullanılır.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split # Veriyi hazırlayıp modelleri test etmek için.

# Makine öğrenmesi modelleri sayısal verilerle çalışır. 
# Yani sınıflarımız (örneğin "kedi", "köpek") önce sayıya, 
# oyunumuzdaki sınıflarımız up, down, right
# sonra bir format olan one-hot vektörlerine dönüştürülmeli.

def onehot_labels(values):
    label_encoder = LabelEncoder()   
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

imgs = glob.glob("img/*.png")

width = 125
height = 50

# boş listeler
X = []
Y = []

# resimleri döngü ile işliyoru
for img in imgs: 
    filename = os.path.basename((img))
    label = filename.split("_")[0]

    # görüntüyü aç, griye dönüştür ve boyutlandır
    im = np.array(Image.open(img).convert("L").resize((width, height)))

    # görüntünin piksel değerlerini 0 ve 1 arasında normalleştir.
    im = im / 255

    # listelere ekle
    X.append(im) # girdi  un, su, tuz
    Y.append(label) # çıktı  ekmek

    
# listeleri array (diziler) çevir
X = np.array(X)

# reshape formülü -> reshape(örnek resim sayısı, genişlik, yükseklik, kanal)
X = X.reshape(X.shape[0], width, height, 1) # girdiler
# çıktı
Y = onehot_labels(Y)

# Veri setini %75 eğitim ve %25 test olarak ayırıyoruz. random_state=2 ile rastgeleliği 
# kontrol ediyoruz ki sonuçlar tekrarlana
# eğitim verisi ve test verisi
train_X, text_X, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=2)


