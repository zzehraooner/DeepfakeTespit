import cv2
import os
import numpy as np
from detector import predict_deepfake # <-- Diğer dosyamızdaki "Beyin"i import ediyoruz!

# --- AYARLAR ---
# Bu script doğrudan çalıştırılırsa bu video kullanılır
VIDEO_PATH = "test_video.mp4"       
TEMP_FRAME_IMAGE = "_temp_frame.jpg" # Kareleri geçici kaydetmek için
FRAME_RATE_SKIP = 30                # Her 30 karede 1 kareyi analiz et

def analyze_video(video_path):
    """
    Bir videoyu kare kare analiz eder ve sonuçları bir sözlük (dict) olarak döndürür.
    """
    
    # Videoyu aç
    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"HATA: '{video_path}' video dosyası açılamadı veya bulunamadı.")
            return {"error": f"'{video_path}' video dosyası açılamadı."}
    except Exception as e:
        return {"error": f"Video dosyası açılırken hata oluştu: {e}"}

    frame_count = 0
    scores = [] # Tespit edilen skorları burada biriktireceğiz

    while True:
        # Videodan bir kare oku
        success, frame = video_capture.read()
        
        # Video bittiyse döngüden çık
        if not success:
            break
            
        # --- ÖRNEKLEME ---
        if frame_count % FRAME_RATE_SKIP == 0:
            print(f"Kare {frame_count} işleniyor...") # Terminalde ilerlemeyi görmek için
            
            # 1. Kareyi geçici bir resim dosyasına yaz
            cv2.imwrite(TEMP_FRAME_IMAGE, frame)
            
            # 2. 'detector.py'deki sihirli fonksiyonumuzu çağır
            score = predict_deepfake(TEMP_FRAME_IMAGE)
            
            # 3. Sonucu listeye ekle (eğer yüz bulunduysa)
            if score is not None:
                scores.append(score)

        frame_count += 1

    # Analiz bittikten sonra temizlik yap
    video_capture.release()
    if os.path.exists(TEMP_FRAME_IMAGE):
        os.remove(TEMP_FRAME_IMAGE)

    # --- FİNAL SONUÇ ---
    if not scores:
        print("\nVideoda hiç yüz tespit edilemedi.")
        return {"error": "Videoda hiç yüz tespit edilemedi."}
        
    # Skorların ortalamasını al
    average_score = np.mean(scores)
    
    # Sonuçları ekrana basmak yerine bir sözlük olarak DÖNDÜR
    result = {
        "file_name": video_path,
        "faces_analyzed": len(scores),
        "average_score": average_score,
        "is_fake": average_score > 0.5
    }
    
    return result

# --- Bu scripti çalıştıralım (Terminalden 'python video_detector.py' yaparsak) ---
if __name__ == "__main__":
    result_data = analyze_video(VIDEO_PATH)
    
    if result_data:
        if "error" in result_data:
            print(result_data["error"])
        else:
            print("\n--- VİDEO ANALİZ SONUCU ---")
            print(f"Analiz edilen dosya: {result_data['file_name']}")
            print(f"Toplam {result_data['faces_analyzed']} adet yüzde analiz yapıldı.")
            print(f"Ortalama Skor: {result_data['average_score']:.4f}")
            
            if result_data['is_fake']:
                print(f"TAHMİN: Bu video BÜYÜK İHTİMALLE SAHTE (Deepfake)")
                print(f"(Ortalama Sahtelik Olasılığı: {result_data['average_score']*100:.2f}%)")
            else:
                print(f"TAHMİN: Bu video BÜYÜK İHTİMALLE GERÇEK")
                print(f"(Ortalama Gerçeklik Olasılığı: {(1-result_data['average_score'])*100:.2f}%)")