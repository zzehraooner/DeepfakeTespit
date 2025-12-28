import streamlit as st
import os
import time
from detector import predict_deepfake
from video_detector import analyze_video # <-- Güncellediğimiz dosyayı import ediyoruz

# --- Geçici dosyaları kaydetmek için bir klasör ---
TEMP_DIR = "temp_uploads"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def save_uploaded_file(uploaded_file):
    """Yüklenen dosyayı geçici bir yola kaydeder ve yolunu döndürür."""
    # Dosyaya benzersiz bir ad ver (çakışmaları önlemek için)
    file_path = os.path.join(TEMP_DIR, f"{int(time.time())}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def display_image_results(score):
    """Resim analiz sonucunu formatlar."""
    if score > 0.5:
        st.error(f"**Tahmin: SAHTE (Deepfake)**")
        st.progress(float(score)) # Skoru 0-1 arasında bir bar olarak göster
        st.metric(label="Sahtelik Olasılığı", value=f"{score*100:.2f}%")
    else:
        st.success(f"**Tahmin: GERÇEK**")
        st.progress(float(score))
        st.metric(label="Sahtelik Olasılığı", value=f"{score*100:.2f}%")

def display_video_results(result_data):
    """Video analiz sonucunu formatlar."""
    if "error" in result_data:
        st.warning(result_data["error"])
        return

    score = result_data['average_score']
    
    if result_data['is_fake']:
        st.error(f"**TAHMİN: Bu video BÜYÜK İHTİMALLE SAHTE (Deepfake)**")
        st.progress(float(score))
        st.metric(label="Ortalama Sahtelik Olasılığı", value=f"{score*100:.2f}%")
    else:
        st.success(f"**TAHMİN: Bu video BÜYÜK İHTİMALLE GERÇEK**")
        st.progress(float(score))
        st.metric(label="Ortalama Sahtelik Olasılığı", value=f"{score*100:.2f}%")
    
    st.info(f"Videoda toplam {result_data['faces_analyzed']} adet yüz analiz edildi.")


# --- Streamlit Arayüzü ---

st.set_page_config(page_title="Deepfake Dedektörü", layout="wide")
st.title("🕵️ Deepfake Tespit Motoru")
st.write("Beraber geliştirdiğimiz deepfake tespit aracının web arayüzü.")

tab1, tab2 = st.tabs(["🖼️ Resim Analizi", "🎬 Video Analizi"])

# --- Resim Analizi Sekmesi ---
with tab1:
    st.header("Tek bir resim karesini analiz edin")
    uploaded_image = st.file_uploader("Analiz için bir resim yükleyin (JPG, PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_image:
        temp_image_path = save_uploaded_file(uploaded_image)
        
        # Resmi göster
        st.image(temp_image_path, caption="Yüklenen Resim", width=400)
        
        if st.button("🖼️ Resmi Analiz Et"):
            with st.spinner("Model yükleniyor ve yüz analiz ediliyor... Lütfen bekleyin."):
                # detector.py'deki fonksiyonu çağır
                score = predict_deepfake(temp_image_path)
            
            if score is not None:
                st.subheader("Analiz Sonucu")
                display_image_results(score)
            else:
                st.error("Resimde analiz edilecek bir yüz bulunamadı.")
            
            # Analizden sonra temp dosyayı sil
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

# --- Video Analizi Sekmesi ---
with tab2:
    st.header("Bir video dosyasını analiz edin")
    uploaded_video = st.file_uploader("Analiz için bir video yükleyin (MP4, MOV)", type=["mp4", "mov"], key="video_uploader")
    
    if uploaded_video:
        temp_video_path = save_uploaded_file(uploaded_video)
        
        # Videoyu göster
        st.video(temp_video_path)
        
        if st.button("🎬 Videoyu Analiz Et"):
            with st.spinner(f"Video analiz ediliyor... Bu işlem videonun uzunluğuna bağlı olarak dakikalar sürebilir. Lütfen bekleyin..."):
                # video_detector.py'deki GÜNCELLENMİŞ fonksiyonu çağır
                result_data = analyze_video(temp_video_path)
            
            if result_data:
                st.subheader("Video Analiz Sonucu")
                display_video_results(result_data)
            else:
                st.error("Video analiz edilirken bir hata oluştu veya hiç yüz bulunamadı.")

            # Analizden sonra temp dosyayı sil
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

st.sidebar.header("Proje Hakkında")
st.sidebar.info(
    "Bu uygulama, bir resim veya videodaki yüzlerin gerçek mi yoksa "
    "deepfake (yapay zeka ile üretilmiş) mi olduğunu tahmin etmek için "
    "Keras/TensorFlow tabanlı bir Evrişimli Sinir Ağı (CNN) modeli kullanır."
)
st.sidebar.warning(
    "**Sorumluluk Reddi:** Bu model, bir eğitim projesi olarak geliştirilmiştir "
    "ve %100 doğruluk garanti etmez. Sonuçlar sadece bilgilendirme amaçlıdır."
)