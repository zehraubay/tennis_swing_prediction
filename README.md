# 🎾 TENİS HAREKET TANIMA VE SINIFLANDIRMA

Bu proje, **bilgisayarla görme (Computer Vision) ve makine öğrenmesi (Machine Learning) tekniklerini kullanarak tenis oyuncusunun hareketlerini tespit eder ve sınıflandırır**. 

## 📌 PROJEDE KULLANILAN TEKNOLOJİLER

### **1️⃣ Mediapipe Pose Estimation (İskelet Takibi)**
- **Kullanılan Teknoloji:** **Mediapipe Pose**
- **Görevi:** Oyuncunun vücut eklemlerini tespit eder (omuz, dirsek, bilek vb.).  
- **Çalışma Mantığı:** 
  - Vücut eklemlerinin **(x, y, z) koordinatlarını çıkarır**.
  - Koordinatlar hareket analizi için kullanılır.  

### **2️⃣ Bounding Box (BBox) ile Oyuncu Algılama**
- **Kullanılan Yöntem:** **Manuel BBox Hesaplama**
- **Görevi:** Vücut koordinatlarından **oyuncunun bulunduğu alanı belirlemek**.
- **Çalışma Mantığı:** 
  - Vücut eklemlerinin en **küçük ve en büyük x, y değerleri** hesaplanarak **dikdörtgen (BBox) çizilir**.

### **3️⃣ XGBoost Makine Öğrenmesi Modeli**
- **Kullanılan Yöntem:** **XGBoost Classifier**
- **Görevi:** Eklem noktalarından alınan verileri **tenis hareketlerine** çevirmek
- **Çalışma Mantığı:**
  - **181 özellikli bir feature vektörü** kullanılarak hareket sınıflandırması yapılır.
  - Model, **Forehand, Backhand, Slice vb. tenis hareketlerini tanır**.

### **4️⃣ OpenCV (cv2) ile Görüntü İşleme**

## 🎥 ÖRNEK ÇIKTI (OUTPUT VIDEO) 🚀
Aşağıda projenin çalıştırılması sonucu elde edilen **örnek video çıkışı** bulunmaktadır:

ffmpeg -i output_with_boxes.mp4 -r 10 -vf "scale=640:-1" output_with_boxes.gif
