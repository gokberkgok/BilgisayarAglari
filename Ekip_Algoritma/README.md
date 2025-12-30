# 🌐 QoS Tabanlı Yönlendirme Algoritmaları Simülasyonu

Bu proje, bilgisayar ağlarında Hizmet Kalitesi (QoS - Quality of Service) gereksinimlerini karşılayan en uygun yolları bulmak amacıyla geliştirilmiş kapsamlı bir simülasyon aracıdır. Modern ve kullanıcı dostu bir arayüz üzerinden, farklı yapay zeka ve optimizasyon algoritmalarını karşılaştırmalı olarak analiz etme imkanı sunar.

## 🚀 Özellikler

*   **Çoklu Algoritma Desteği:** 6 farklı optimizasyon algoritması (SARSA, Genetik, ACO, Q-Learning, PSO, VNS) ile yol hesaplama.
*   **Gelişmiş Görselleştirme:** Ağ topolojisinin `NetworkX` ve `Matplotlib` tabanlı interaktif görselleştirmesi.
*   **QoS Analizi:** Gecikme (Delay), Güvenilirlik (Reliability) ve Bant Genişliği (Bandwidth) gibi metriklerin detaylı analizi.
*   **Toplu Deney Modu:** DemandData.CSV dosyasından otomatik olarak test etme, duraklama/devam ettirme ve sonuçları raporlama.
*   **Kullanıcı Kontrollü Seed:** Arayüzden seed değerini manuel olarak ayarlama veya rastgele çalıştırma seçeneği.
*   **Tekrarlanabilirlik:** Tüm algoritmalar için `seed` (tohum) desteği sayesinde %100 tekrarlanabilir ve doğrulanabilir sonuçlar.

## 🧠 Algoritmalar ve Katkıda Bulunanlar
1.  **SARSA Algoritması** - *Oguzhan Demirbas*  https://github.com/OguzIronCode
2.  **Genetik Algoritma (GA)** - *Azra Kaya* https://github.com/kayazra
3.  **Karınca Kolonisi Optimizasyonu (ACO)** - *Aivaz Arysbay* https://github.com/Aivazz
4.  **Q-Learning** - *Gokberk Gok* https://github.com/gokberkgok
5.  **Parçacık Sürüsü Optimizasyonu (PSO)** - *Salim Caner* https://github.com/canerozal
6.  **Değişken Komşuluk Arama (VNS)** - *Yigit Emre* https://github.com/yigitemre22
7.  **Arayüz** 
        *Enes Kuru* - https://github.com/eneskru
        *Umut Kağan Ceylan* - https://github.com/umutkaganc
        *Wala Quasem* - https://github.com/wala127

## 🛠️ Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Repoyu Klonlayın:**
    ```bash
    git clone https://github.com/gokberkgok/BilgisayarAglari
    cd Ekip-Algoritma
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
    *   Kaynak ve Hedef düğümleri seçin (0-249 arası).
    *   Minimum Bant Genişliği ve QoS ağırlıklarını (Gecikme, Güvenilirlik, Kaynak) ayarlayın.
    *   **Seed Kontrolü:** 
        *   "Sabit Seed Kullan" kutusunu işaretleyerek belirli bir seed değeri ile tekrarlanabilir sonuçlar elde edebilirsiniz.
        *   Kutu işaretli değilse, her çalıştırmada farklı rastgele sonuçlar üretilir.
    *   İstediğiniz algoritmayı seçip **"HESAPLA ve GÖSTER"** butonuna tıklayın.
    *   Sonuçlar, yol animasyonu ve detaylı metrikler ekranın sağ tarafında gösterilecektir.

*   **📊 Toplu Deney:**
    *   Bu sekmede, `DemandData.csv` dosyasındaki tüm senaryolar sırasıyla test edilir.
    *   Algoritmayı seçin ve **"🧪 TESTİ BAŞLAT"** butonuna basın.
    *   Test sırasında **"⏸️ DURAKLAT"** butonu ile testi duraklatabilir, **"▶️ DEVAM ET"** ile devam ettirebilirsiniz.
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

### Kullanıcı Kontrollü Seed

Arayüzün **Tekli Analiz** sekmesinde, kullanıcılar seed değerini manuel olarak kontrol edebilir:

*   **"Sabit Seed Kullan" Checkbox:** Bu kutu işaretlendiğinde, yanındaki spinbox'tan belirlenen seed değeri kullanılır.
*   **Seed Değeri Spinbox:** 0-9999 arası bir seed değeri seçilebilir (varsayılan: 42).
*   **Rastgele Mod:** Checkbox işaretli değilse, algoritmalar `seed=None` ile çalışır ve her çalıştırmada farklı sonuçlar üretir.

Bu özellik sayesinde:
*   Aynı parametrelerle yapılan testlerin aynı sonuçları vermesi garanti edilir (sabit seed ile)
*   Farklı çözüm uzaylarını keşfetmek için rastgele mod kullanılabilir
*   Sonuçların doğrulanabilirliği ve tekrarlanabilirliği sağlanır


### Varsayılan Seed Değeri

Uygulama, varsayılan olarak **seed=42** değerini kullanır. Bu değer:
*   Tüm algoritmalarda tutarlı sonuçlar üretir
*   Aynı parametrelerle yapılan testlerin aynı sonuçları vermesini garanti eder
*   GitHub ve akademik paylaşımlar için sonuçların doğrulanabilir olmasını sağlar

### Doğrulama

Seed implementasyonunu test etmek için `verify_seed.py` dosyası kullanılabilir:

```bash
python verify_seed.py
```

Bu script, aynı seed değeri ile yapılan iki çalıştırmanın özdeş sonuçlar verdiğini, farklı seed değerleri ile yapılan çalıştırmaların ise farklı sonuçlar ürettiğini doğrular.

---
*Bu proje BSM307/317 Dönem Projesi kapsamında geliştirilmiştir.*
