import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

# Yüz dedektörünü (MTCNN) başlat
detector = MTCNN()

def extract_face(image_path, required_size=(112, 112), margin_percent=20):
    """
    Bir görüntüden yüzü bulan, kırpan, marjin ekleyen ve yeniden boyutlandıran fonksiyon.
    
    Arguments:
        image_path (str): Görüntü dosyasının yolu.
        required_size (tuple): Modelin beklediği çıktı boyutu (112, 112).
        margin_percent (int): Kırpılan yüze eklenecek marjin yüzdesi.
    
    Returns:
        numpy.ndarray: Yeniden boyutlandırılmış yüz görüntüsü veya None (yüz bulunamazsa).
    """
    
    # Görüntüyü oku
    image = cv2.imread(image_path)
    
    # Görüntü okunamadıysa (örn. dosya yolu yanlışsa)
    if image is None:
        print(f"HATA: '{image_path}' dosyası okunamadı veya bozuk.")
        return None
        
    # MTCNN, BGR değil RGB formatında görüntü bekler
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Yüzleri tespit et
    results = detector.detect_faces(image_rgb)
    
    if not results:
        print(f"Uyarı: '{image_path}' içinde yüz bulunamadı.")
        return None

    # En büyük yüzü bul (güven skoruna göre)
    primary_face = max(results, key=lambda face: face['confidence'])
    
    # Yüzün koordinatlarını al
    x1, y1, width, height = primary_face['box']
    
    # Koordinatların negatif olmamasını sağla
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # Marjin hesapla (yüzün biraz etrafını da alalım)
    margin_w = int(width * margin_percent / 100)
    margin_h = int(height * margin_percent / 100)
    
    # Marjinli yeni koordinatlar
    x1_margin = max(0, x1 - margin_w)
    y1_margin = max(0, y1 - margin_h)
    x2_margin = min(image.shape[1], x2 + margin_w)
    y2_margin = min(image.shape[0], y2 + margin_h)
    
    # Yüzü kırp (EKSİK OLAN SATIR BUYDU)
    cropped_face = image[y1_margin:y2_margin, x1_margin:x2_margin]

    # Kırpılan yüz boşsa (örn. koordinatlar hatalıysa)
    if cropped_face.size == 0:
        print(f"HATA: Yüz koordinatları hatalı, kırpma işlemi başarısız oldu.")
        return None
        
    # Gerekli boyuta yeniden boyutlandır
    resized_face = cv2.resize(cropped_face, required_size, interpolation=cv2.INTER_AREA)
    
    return resized_face

# --- Bu fonksiyonu test etmek için ---
# (Bu scripti doğrudan çalıştırırsanız 'test_image.jpg'yi test eder)
if __name__ == "__main__":
    face_image = extract_face('test_image.jpg')
    
    if face_image is not None:
        print(f"Yüz başarıyla bulundu ve {face_image.shape} boyutuna getirildi.")
        # Kırpılan yüzü 'cropped_face.jpg' olarak kaydet
        cv2.imwrite('cropped_face.jpg', face_image)
        print("Kırpılan yüz 'cropped_face.jpg' olarak kaydedildi.")
    else:
        print("Test görüntüsünde yüz bulunamadı.")