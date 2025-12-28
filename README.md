🤖 Deepfake Detection System | Deepfake Tespit Sistemi
Bu proje, yapay zeka ve derin öğrenme tekniklerini kullanarak, dijital içeriklerdeki (görsel ve video) manipülasyonları ve sahte yüz değişimlerini tespit etmek amacıyla geliştirilmiştir. Bilgi kirliliği ve dijital sahtecilikle mücadelede güvenilir bir analiz aracı olmayı hedefler.

🚀 Proje Hakkında
Dijital dünyada "Deepfake" içeriklerin artmasıyla birlikte, gerçeği sahteden ayırt etmek her geçen gün zorlaşıyor. Bu sistem:

Frame-by-Frame Analiz: Videoları karelere bölerek her bir saniyedeki anomaliyi yakalar.

CNN & LSTM Mimarisi: Görsel özellikleri yakalamak için Konvolüsyonel Sinir Ağları (CNN) ve zaman içindeki tutarsızlıkları belirlemek için LSTM katmanlarını kullanır.

Yüz İşaretleyici (Landmark) Analizi: Göz kırpma frekansı, ağız hareketleri ve cilt dokusundaki yapaylıkları denetler.

🛠️ Teknik Altyapı
Dil: Python 3.10+

Derin Öğrenme: TensorFlow / Keras / PyTorch

Görüntü İşleme: OpenCV (Open Source Computer Vision Library)

Veri Analizi: NumPy, Pandas, Matplotlib

📦 Kurulum ve Çalıştırma
1. Repoyu bilgisayarınıza çekin:
git clone https://github.com/zzehraooner/DeepfakeTespit.git
cd DeepfakeTespit
2. Gerekli bağımlılıkları yükleyin:
pip install -r requirements.txt
3. Sistemi başlatın:
python main.py --input path/to/your/video.mp4

📊 Hedeflenen Sonuçlar
Sistem, analiz edilen içerik için bir "Güven Skoru" (Confidence Score) üretir:

%0-30: Muhtemelen Gerçek

%30-70: Şüpheli İçerik

%70-100: Yüksek Olasılıklı Deepfake

Not: Bu proje geliştirilme aşamasındadır. Katkıda bulunmak isterseniz lütfen bir Pull Request açın veya bir Issue bildirin.
