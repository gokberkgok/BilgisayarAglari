# 🌐 QoS Tabanlı Yönlendirme Algoritmaları Simülasyonu

Bu proje, bilgisayar ağlarında Hizmet Kalitesi (QoS - Quality of Service) gereksinimlerini karşılayan en uygun yolları bulmak amacıyla geliştirilmiş kapsamlı bir simülasyon aracıdır. Modern ve kullanıcı dostu bir arayüz üzerinden, farklı yapay zeka ve optimizasyon algoritmalarını karşılaştırmalı olarak analiz etme imkanı sunar.

## 🚀 Özellikler

*   **Çoklu Algoritma Desteği:** 6 farklı optimizasyon algoritması (SARSA, Genetik, ACO, Q-Learning, PSO, VNS) ile yol hesaplama.
*   **Gelişmiş Görselleştirme:** Ağ topolojisinin `NetworkX` ve `Matplotlib` tabanlı interaktif görselleştirmesi.
*   **QoS Analizi:** Gecikme (Delay), Güvenilirlik (Reliability) ve Bant Genişliği (Bandwidth) gibi metriklerin detaylı analizi.
*   **Modern Arayüz:** PyQt6 ile geliştirilmiş, Neon/Cyberpunk temalı, kullanımı kolay grafik arayüz (GUI).
*   **Toplu Deney Modu:** CSV dosyalarından yüklenen yüzlerce senaryoyu otomatik olarak test etme ve sonuçları raporlama.
*   **Tekrarlanabilirlik:** Tüm algoritmalar için `seed` (tohum) desteği sayesinde %100 tekrarlanabilir ve doğrulanabilir sonuçlar.

## 🧠 Algoritmalar ve Katkıda Bulunanlar

Proje kapsamında aşağıdaki algoritmalar implemente edilmiştir:

1.  **SARSA Algoritması** - *Oguzhan Demirbas*
2.  **Genetik Algoritma (GA)** - *Azra Kaya*
3.  **Karınca Kolonisi Optimizasyonu (ACO)** - *Aivaz Arysbay*
4.  **Q-Learning** - *Gokberk Gok*
5.  **Parçacık Sürüsü Optimizasyonu (PSO)** - *Salim Caner*
6.  **Değişken Komşuluk Arama (VNS)** - *Yigit Emre*

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone https://github.com/kullaniciadi/proje-adi.git
    cd proje-adi
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Kullanım

Uygulamayı başlatmak için ana Python dosyasını çalıştırın:

```bash
python Arayuz.py
```

### Arayüz Sekmeleri

*   **🔍 Tekli Analiz:**
    *   Kaynak ve Hedef düğümleri seçin.
    *   Minimum Bant Genişliği ve QoS ağırlıklarını (Gecikme, Güvenilirlik, Kaynak) ayarlayın.
    *   İstediğiniz algoritmayı seçip **"HESAPLA ve GÖSTER"** butonuna tıklayın.
    *   Sonuçlar, yol animasyonu ve detaylı metrikler ekranın sağ tarafında gösterilecektir.

*   **📊 Toplu Deney:**
    *   Bu sekmede, `DemandData.csv` dosyasındaki tüm senaryolar sırasıyla test edilir.
    *   Algoritmayı seçin ve **"🧪 TESTİ BAŞLAT"** butonuna basın.
    *   Sonuçlar anlık olarak tabloda listelenir ve dilerseniz CSV/Excel formatında kaydedilebilir.

## 📂 Dosya Yapısı

*   `Arayuz.py`: Ana uygulama ve GUI kodu.
*   `Sarsa_Algoritmasi_*.py`: SARSA algoritması implementasyonu.
*   `Genetik_Algoritmasi_*.py`: Genetik algoritma implementasyonu.
*   `Karınca_Kolonisi_*.py`: ACO ve alternatif GA implementasyonu.
*   `Q_Learning_*.py`: Q-Learning algoritması implementasyonu.
*   `Parcacık_Surusu_*.py`: PSO implementasyonu.
*   `VNS_Algorithm_*.py`: VNS implementasyonu.
*   `*.csv`: Ağ topolojisi (Node/Edge) ve talep verileri.

## 📝 Notlar

*   Ağ topolojisi, proje dizinindeki CSV dosyalarından otomatik olarak oluşturulur. CSV dosyaları bulunamazsa rastgele bir Watts-Strogatz grafı üretilir.
*   Tekrarlanabilirlik için tüm rastgele sayı üreteçleri sabit bir `seed` (varsayılan: 42) kullanacak şekilde yapılandırılmıştır.

---
*Bu proje BSM307/317 Dönem Projesi kapsamında geliştirilmiştir.*
