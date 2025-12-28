import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from face_extractor import extract_face

def build_xception():
    base = tf.keras.applications.Xception(weights=None, input_shape=(299,299,3), include_top=False)
    x = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=Dense(1, activation='sigmoid')(x))

def build_mesonet():
    x_in = Input(shape=(256, 256, 3))
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x_in)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=x_in, outputs=x)

def predict_deepfake(image_path):
    # Xception'ın beklediği 299x299 boyutunda yüzü alıyoruz
    face_raw = extract_face(image_path, required_size=(299, 299))
    if face_raw is None: 
        return None

    scores = []

    # 1. Model: Xception Analizi
    if os.path.exists('model_xception.h5'):
        try:
            m1 = build_xception()
            m1.load_weights('model_xception.h5')
            # Normalizasyon: [-1, 1] aralığı
            face_x = (face_raw.astype('float32') / 127.5) - 1.0
            s1 = m1.predict(np.expand_dims(face_x, 0), verbose=0)[0][0]
            scores.append(s1)
            print(f"Xception Analiz Sonucu: {s1:.4f}")
        except Exception as e:
            print(f"⚠️ Xception Çalıştırılamadı: {e}")

    # 2. Model: MesoNet Analizi
    if os.path.exists('model_meso.h5'):
        try:
            m2 = build_mesonet()
            m2.load_weights('model_meso.h5')
            # MesoNet için 299x299 -> 256x256 yeniden boyutlandırma
            face_m = cv2.resize(face_raw, (256, 256))
            # Normalizasyon: [0, 1] aralığı
            face_m = face_m.astype('float32') / 255.0
            s2 = m2.predict(np.expand_dims(face_m, 0), verbose=0)[0][0]
            scores.append(s2)
            print(f"MesoNet Analiz Sonucu: {s2:.4f}")
        except Exception as e:
            print(f"⚠️ MesoNet Çalıştırılamadı: {e}")

    if not scores:
        return None

    # Ensemble: İki modelin ortalamasını al
    final_score = np.mean(scores)
    return final_score