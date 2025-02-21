# Öğreneceklerimiz

# modelimizi inşa edeceğiz 

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
width = 250
height = 100

# boş listeler
X = []
Y = []

# resimleri döngü ile işliyoruz
for img in imgs: 
    filename = os.path.basename(img)
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
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=2) # random.randint(112, 1112)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1))) 
# İlk Convolution katmanını ekliyoruz. 
# 32: 32 farklı filtre kullanılacak.
# kernel_size=(3, 3): Filtrelerin boyutu 3x3 piksel olacak.
# activation='relu': Aktivasyon fonksiyonu olarak ReLU kullanılacak.
# relu fonk görevi modelin öğrenme sırasında negatif değerleri sıfırlayıp pozitif değerleri olduğu gibi kullanması sağlanacak.
# input_shape=(width, height, 1): Giriş görüntüsünün boyutu (genişlik, yükseklik) ve renk kanalı sayısı (1 = gri tonlama).

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) 
# İkinci Convolution katmanını ekliyoruz. 
# 64: 64 farklı filtre kullanılacak.
# kernel_size=(3, 3): Filtrelerin boyutu 3x3 piksel olacak.
# activation='relu': Aktivasyon fonksiyonu olarak ReLU kullanılacak.

model.add(MaxPooling2D(pool_size=(2, 2)))  # Max Pooling katmanı ekleniyor.
# pool_size=(2, 2): 2x2 piksellik bölgelerdeki en büyük değeri alarak boyutu küçültecek.

model.add(Dropout(0.25))  # Dropout katmanı ekleniyor.
# 0.25: Nöronların %25'i eğitim sırasında rastgele olarak devre dışı bırakılacak (overfitting'i engellemek için).

model.add(Flatten())  # Flatten katmanı ekleniyor.
# Evrişim katmanlarından çıkan çok boyutlu veriyi tek boyutlu bir vektöre dönüştürecek.

model.add(Dense(128, activation='relu'))  # Tamamen bağlı (Dense) katman ekleniyor.
# 128: 128 nöron içerecek.
# activation='relu': Aktivasyon fonksiyonu olarak ReLU kullanılacak.

model.add(Dropout(0.4))  # Dropout katmanı tekrar ekleniyor.
# 0.4: Nöronların %40'ı eğitim sırasında rastgele olarak devre dışı bırakılacak.

model.add(Dense(2, activation='softmax'))  # Çıkış katmanı ekleniyor.
# 3: 3 sınıf var.
# activation='softmax': Softmax aktivasyon fonksiyonu, olasılık dağılımı üretecek.

if os.path.exists("./trex_weight.h5"):  # Eğer "trex_weight.h5" adlı dosya bu konumda varsa:
    model.load_weights("trex_weight.h5")  # Modelin ağırlıklarını bu dosyadan yükle.
    print("Ağırlık dosyası yüklendi.")  # Kullanıcıya bilgi ver.


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
# Modeli derle.   optimizer='SGD'
# loss: Kayıp fonksiyonu olarak kategorik çapraz entropi kullanılıyor (çok sınıflı sınıflandırma için).
# optimizer: Optimizasyon algoritması olarak Adam kullanılıyor.
# metrics: Başarı metriği olarak doğruluk (accuracy) kullanılıyor.

model.fit(train_X, train_y, epochs=35, batch_size=64) # 20
# Modeli eğit.
# train_X: Eğitim verileri (giriş).
# train_y: Eğitim etiketleri (çıkış).
# epochs: Eğitim boyunca tüm verilerin kaç kez taranacağı (35 defa).
# batch_size: Her bir adımda kaç örnek işleneceği (64 örnek).

print(model.summary())  # Modelin özetini (katmanlar, parametre sayıları vb.) ekrana yazdır.

scores = model.evaluate(train_X, train_y)  # Eğitim verileriyle modelin performansını değerlendir.
print(f"Train: {model.metrics_names[1]} : {scores[1]*100}")  # Eğitim doğruluğunu ekrana yazdır.

scores = model.evaluate(test_X, test_y)  # Test verileriyle modelin performansını değerlendir.
print(f"Test: {model.metrics_names[1]} : {scores[1]*100}")  # Test doğruluğunu ekrana yazdır.

open("model.json","w").write(model.to_json())  # Modelin mimarisini JSON dosyasına kaydet.
model.save_weights("trex_weight.h5")  # Modelin ağırlıklarını H5 dosyasına kaydet.
