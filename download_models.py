import urllib.request
import os
import ssl

def download_with_fallbacks(filenames_and_urls):
    # macOS SSL hatasını çözmek için context oluştur
    context = ssl._create_unverified_context()
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for filename, urls in filenames_and_urls.items():
        success = False
        print(f"--- {filename} İÇİN İNDİRME DENENİYOR ---")
        for url in urls:
            try:
                print(f"Denenen Link: {url}")
                urllib.request.urlretrieve(url, filename)
                # Dosya boyutunu kontrol et (boş veya 404 sayfası mı?)
                if os.path.getsize(filename) > 1000: # 1KB'den büyükse başarılıdır
                    print(f"✅ BAŞARILI: {filename} indirildi.\n")
                    success = True
                    break
                else:
                    os.remove(filename)
            except Exception as e:
                print(f"⚠️ Bu link başarısız oldu: {e}")
        
        if not success:
            print(f"❌ HATA: {filename} hiçbir kaynaktan indirilemedi.\n")

# Denenecek alternatif linkler (Main vs Master farkları için)
models_to_download = {
    "model_xception.h5": [
        "https://github.com/KrishnaPrasanna21/Deepfake_Video_Detection/raw/main/deepfake-detection-model.h5"
    ]
}
    

if __name__ == "__main__":
    download_with_fallbacks(models_to_download)