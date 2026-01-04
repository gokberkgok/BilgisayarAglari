# 🌐 QoS Tabanlı Yönlendirme Algoritmaları Simülasyonu

Bu proje, bilgisayar ağlarında Hizmet Kalitesi (QoS - Quality of Service) gereksinimlerini karşılayan en uygun yolları bulmak amacıyla geliştirilmiş kapsamlı bir simülasyon aracıdır. Modern ve kullanıcı dostu bir arayüz üzerinden, farklı yapay zeka ve optimizasyon algoritmalarını karşılaştırmalı olarak analiz etme imkanı sunar.

## 🚀 Özellikler

*   **Çoklu Algoritma Desteği:** 6 farklı optimizasyon algoritması (SARSA, Genetik, ACO, Q-Learning, PSO, VNS) ile yol hesaplama.
*   **Gelişmiş Görselleştirme:** Ağ topolojisinin `NetworkX` ve `Matplotlib` tabanlı interaktif görselleştirmesi.
*   **QoS Analizi:** Gecikme (Delay), Güvenilirlik (Reliability) ve Bant Genişliği (Bandwidth) gibi metriklerin detaylı analizi.
*   **Modern Arayüz:** PyQt6 ile geliştirilmiş, Neon/Cyberpunk temalı, kullanımı kolay grafik arayüz (GUI).
*   **Toplu Deney Modu:** CSV dosyalarından yüklenen yüzlerce senaryoyu otomatik olarak test etme, duraklama/devam ettirme ve sonuçları raporlama.
*   **Kullanıcı Kontrollü Seed:** Arayüzden seed değerini manuel olarak ayarlama veya rastgele çalıştırma seçeneği.
*   **Tekrarlanabilirlik:** Tüm algoritmalar için `seed` (tohum) desteği sayesinde %100 tekrarlanabilir ve doğrulanabilir sonuçlar.
*   **Standart Node İndeksleme:** Tüm projede 0-249 arası tutarlı düğüm numaralandırması.

## 🧠 Algoritmalar ve Katkıda Bulunanlar

Proje kapsamında aşağıdaki algoritmalar implemente edilmiştir:

1.  **SARSA Algoritması** - *Oguzhan Demirbas*
2.  **Genetik Algoritma (GA)** - *Azra Kaya*
3.  **Karınca Kolonisi Optimizasyonu (ACO)** - *Aivaz Arysbay*
4.  **Q-Learning** - *Gokberk Gok*
5.  **Parçacık Sürüsü Optimizasyonu (PSO)** - *Salim Caner*
6.  **Değişken Komşuluk Arama (VNS)** - *Yigit Emre*
7.  **ARAYÜZ** - *Enes Kuru-Umut Kağan-Ceylan-Wala Quasem*


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
    *   Kaynak ve Hedef düğümleri seçin (0-249 arası).
    *   Minimum Bant Genişliği ve QoS ağırlıklarını (Gecikme, Güvenilirlik, Kaynak) ayarlayın.
    *   **Seed Kontrolü:** 
        *   "Sabit Seed Kullan" kutusunu işaretleyerek belirli bir seed değeri ile tekrarlanabilir sonuçlar elde edebilirsiniz.
        *   Kutu işaretli değilse, her çalıştırmada farklı rastgele sonuçlar üretilir.
    *   İstediğiniz algoritmayı seçip **"HESAPLA ve GÖSTER"** butonuna tıklayın.
    *   Sonuçlar, yol animasyonu ve detaylı metrikler ekranın sağ tarafında gösterilecektir.

*   **📊 Toplu Deney:**
    *   Bu sekmede, `DemandData.csv` dosyasındaki tüm senaryolar sırasıyla test edilir.
    *   Test sırasında **"⏸️ DURAKLAT"** butonu ile testi duraklatabilir, **"▶️ DEVAM ET"** ile devam ettirebilirsiniz.
    *   Sonuçlar anlık olarak tabloda listelenir ve dilerseniz CSV/Excel formatında kaydedilebilir.

*   **⚖️ Algoritma Karşılaştırma:**
    *   Bu sekmede, 6 farklı algoritmayı (SARSA, Q-Learning, GA, ACO, PSO, VNS) aynı senaryo üzerinde karşılaştırabilirsiniz.
    *   Kaynak, Hedef ve Bant Genişliği değerlerini girdikten sonra **"🚀 KARŞILAŞTIR"** butonuna basmanız yeterlidir.
    *   Tabloda her algoritmanın **Maliyet**, **Süre**, **Yol Uzunluğu** ve **Durum** bilgileri karşılaştırmalı olarak gösterilir.
    *   Sonuçları CSV formatında dışa aktarabilirsiniz.

## 📂 Dosya Yapısı

*   `Arayuz.py`: Ana uygulama ve GUI kodu.
*   `Sarsa_Algoritmasi_*.py`: SARSA algoritması implementasyonu.
*   `Genetik_Algoritmasi_*.py`: Genetik algoritma implementasyonu.
*   `Karınca_Kolonisi_*.py`: ACO ve alternatif GA implementasyonu.
*   `Q_Learning_*.py`: Q-Learning algoritması implementasyonu.
*   `Parcacık_Surusu_*.py`: PSO implementasyonu.
*   `VNS_Algorithm_*.py`: VNS implementasyonu.
*   `*.csv`: Ağ topolojisi (Node/Edge) ve talep verileri.

## 🔬 Tekrarlanabilirlik ve Seed Desteği

Proje, bilimsel araştırma ve akademik çalışmalar için kritik öneme sahip **%100 tekrarlanabilir sonuçlar** sunmaktadır. Tüm algoritmalar `seed` (rastgele sayı üreteci tohum değeri) parametresi ile çalışacak şekilde güncellenmiştir.

### Kullanıcı Kontrollü Seed

Arayüzün **Tekli Analiz** sekmesinde, kullanıcılar seed değerini manuel olarak kontrol edebilir:

*   **"Sabit Seed Kullan" Checkbox:** Bu kutu işaretlendiğinde, yanındaki spinbox'tan belirlenen seed değeri kullanılır.
*   **Seed Değeri Spinbox:** 0-9999 arası bir seed değeri seçilebilir (varsayılan: 42).
*   **Rastgele Mod:** Checkbox işaretli değilse, algoritmalar `seed=None` ile çalışır ve her çalıştırmada farklı sonuçlar üretir.

Bu özellik sayesinde:
*   Aynı parametrelerle yapılan testlerin aynı sonuçları vermesi garanti edilir (sabit seed ile)
*   Farklı çözüm uzaylarını keşfetmek için rastgele mod kullanılabilir
*   Sonuçların doğrulanabilirliği ve tekrarlanabilirliği sağlanır

### Seed Implementasyonu

Her algoritma dosyası, `seed` parametresini kabul eder ve aşağıdaki şekilde kullanır:

*   **SARSA** (`Sarsa_Algoritmasi_Oguzhan_Demirbas.py`): `sarsa_route(G, S, D, min_bw, episodes, seed=42)`
*   **Genetik Algoritma** (`Genetik_Algoritmasi_Azra_Kaya.py`): `genetic_algorithm(..., seed=42)`
*   **ACO** (`Karınca_Kolonisi_Algoritmasi_Aivaz_Arysbay.py`): `ACO.solve(..., seed=42)`
*   **Q-Learning** (`Q_Learning_Gokberk_Gok_.py`): `train_q_learning(..., seed=42)`
*   **PSO** (`Parcacık_Surusu_Optimizasyonu_Salim_Caner.py`): `PSO(..., seed=42)`
*   **VNS** (`VNS_Algorithm_Yigit_Emre.py`): `VNS.run(..., seed=42)`

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

## 🆕 Son Güncellemeler

### Versiyon 2.0 - Aralık 2025

*   **✅ Kullanıcı Kontrollü Seed:** Arayüze "Sabit Seed Kullan" checkbox ve seed değeri spinbox eklendi. Kullanıcılar artık manuel olarak seed değerini kontrol edebilir veya rastgele mod kullanabilir.
*   **✅ Duraklat/Devam Et:** Toplu deney testlerinde duraklama ve devam ettirme özelliği eklendi.
*   **✅ Node ID Standardizasyonu:** Tüm projede 0-249 arası tutarlı düğüm numaralandırması sağlandı.
*   **✅ Kapsamlı Dokümantasyon:** Tüm algoritma dosyalarına satır satır açıklayıcı yorumlar eklendi.
*   **✅ Yol Görselleştirmesi:** Algoritma sonuçlarında bulunan yolun düğüm sıralaması ile gösterimi eklendi.
*   **✅ Algoritma Karşılaştırma:** Tüm algoritmaları aynı anda çalıştırıp performanslarını (sürei maliyet, hop) kıyaslayan yeni bir sekme eklendi.
*   **✅ Gelişmiş UI/UX:** Neon/Cyberpunk temalı modern arayüz tasarımı ve kullanıcı deneyimi iyileştirmeleri.



----------------------------------------------------------------------------------------------



# 🌐 1000 Node Arayüzü: Büyük Ölçekli Ağ Simülasyonu

Gelişmiş optimizasyon algoritmaları kullanarak QoS odaklı çok amaçlı yönlendirme için yüksek performanslı bir ağ simülasyon çerçevesi.

## 📋 Genel Bakış

Bu proje, büyük ölçekli ağları (1000+ düğüme kadar) simüle eder ve Hizmet Kalitesi (QoS) metriklerine dayalı en uygun yolları bulmak için birden fazla yönlendirme algoritması uygular:

-   **Gecikme (Delay)** (latency)
-   **Güvenilirlik (Reliability)** (packet loss)
-   **Bant Genişliği (Bandwidth)** (throughput)

## ✨ Özellikler

-   🎯 **Çok Amaçlı Optimizasyon:** Birden fazla QoS metriğini aynı anda dengeleyin
-   🐜 **Karınca Kolonisi Optimizasyonu (ACO):** Biyo-ilhamlı meta-sezgisel algoritma
-   📊 **Tamsayı Doğrusal Programlama (ILP):** Kesin optimizasyon temel çizgisi
-   📈 **Pareto Analizi:** Hedefler arasındaki takasları (trade-offs) keşfedin
-   🎨 **Modern Arayüz:** PyQt6 tabanlı, göze hoş gelen karanlık tema
-   🔄 **Tekrarlanabilir Sonuçlar:** Seed (tohum) tabanlı rastgele üretim
-   ⚡ **Yüksek Performans:** Büyük ölçekli ağlar için optimize edilmiştir

## 🚀 Hızlı Başlangıç

### Gereksinimler

-   Python 3.8+
-   pip paket yöneticisi

### Kurulum

```bash
# Depoyu klonlayın
cd network_simulation

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### Uygulamayı Çalıştırma

```bash
# Proje kök dizininden
python -m network_simulation.main

# Veya doğrudan
python network_simulation/main.py
```

## 🎮 Kullanım Kılavuzu

### 1. Ağ Topolojisi Oluşturun

1.  **Nodes (Düğüm)** sayısını ayarlayın (varsayılan: 1000)
2.  Kenar oluşturma **Probability (Olasılık)** değerini ayarlayın (varsayılan: 0.4)
3.  **GENERATE NETWORK** butonuna tıklayın

Sistem, rastgele yönlendirilmiş bir graf oluşturmak için Erdős-Rényi G(n,p) modelini kullanır.

### 2. Simülasyonu Yapılandırın

-   **Source ID:** Başlangıç düğümü (0 tabanlı indeksleme)
-   **Dest ID:** Hedef düğüm
-   **Weights (Ağırlıklar):** Her metriğin göreceli önemi
    -   `w_delay`: Gecikme ağırlığı (0.0 - 1.0)
    -   `w_rel`: Güvenilirlik ağırlığı (0.0 - 1.0)
    -   `w_res`: Bant genişliği ağırlığı (0.0 - 1.0)

### 3. Algoritmaları Çalıştırın

-   **ACO:** Hızlı, ölçeklenebilir meta-sezgisel (büyük ağlar için önerilir)
-   **ILP:** Kesin çözüm (büyük ağlarda zaman aşımına uğrayabilir)
-   **Pareto:** Çok amaçlı analiz (takas uzayını keşfeder)

### 4. Sonuçları Görselleştirin

-   **Network Visualizer:** Vurgulanmış yollarla interaktif grafik
-   **Pareto Analysis:** Baskın olmayan çözümlerin 3B dağılım grafiği

## 🔧 Algoritma Parametreleri

### ACO (Karınca Kolonisi Optimizasyonu)

```python
DEFAULT_NUM_ANTS = 20           # İterasyon başına karınca sayısı
DEFAULT_MAX_ITERATIONS = 50     # Maksimum iterasyon
DEFAULT_ALPHA = 1.0             # Feromon önemi
DEFAULT_BETA = 2.0              # Sezgisel önem
DEFAULT_RHO = 0.1               # Buharlaşma oranı
DEFAULT_Q0 = 0.9                # Sömürü (Exploitation) olasılığı
```

### ILP (Tamsayı Doğrusal Programlama)

```python
time_limit = 30  # Maksimum çözüm süresi (saniye)
```

## 📊 Ağ Metrikleri

### Kenar Metrikleri
-   **Link Delay (Bağlantı Gecikmesi):** 2-20 ms (tekdüze dağılım)
-   **Reliability (Güvenilirlik):** 0.95-0.9999 (tekdüze dağılım)
-   **Bandwidth (Bant Genişliği):** 100-10000 Mbps (tekdüze dağılım)

### Düğüm Metrikleri
-   **Processing Delay (İşleme Gecikmesi):** Düğüm başına 1-5 ms

## 🏗️ Proje Yapısı

```
network_simulation/
├── core/
│   ├── algorithms.py      # ACO, ILP, Pareto implementasyonları
│   └── network_model.py   # Ağ grafı oluşturma
├── ui/
│   └── gui.py            # PyQt6 arayüzü
├── main.py               # Uygulama giriş noktası
├── verify_algorithms.py  # Birim testleri
└── requirements.txt      # Bağımlılıklar
```

## 🧪 Test Etme

Doğrulama testlerini çalıştırın:

```bash
python -m network_simulation.verify_algorithms
```

Testler şunları içerir:
-   Küçük graf doğrulaması (N=50)
-   Pareto analizi doğruluğu
-   Büyük graf kıyaslaması (N=1000)

## 📈 Performans

| Ağ Boyutu | ACO Süresi | ILP Süresi |
|-----------|------------|------------|
| 50 düğüm  | ~0.5s      | ~2s        |
| 250 düğüm | ~3s        | ~15s       |
| 1000 düğüm| ~12s       | zaman aşımı|

*Intel i7-10700K, 16GB RAM üzerindeki kıyaslamalar*

## 🎨 Arayüz Özellikleri

-   **Karanlık Tema:** Modern, göz dostu arayüz
-   **Yakınlaştırma Kontrolleri:** Büyük ağlarda kolayca gezinin
-   **Gerçek Zamanlı Günlükler:** Algoritma ilerlemesini izleyin
-   **Yol Vurgulama:** Çözümler için görsel geri bildirim
-   **Derece Tabanlı Stil:** Düğüm boyutu/rengi bağlantıyı yansıtır

## 🔬 Algoritma Detayları

### ACO Maliyet Fonksiyonu

Skalerleştirilmiş maliyet, normalleştirilmiş metrikleri birleştirir:

```
Cost = w_delay × (delay/25000) + 
       w_rel × (-log(reliability)/50) + 
       w_res × (1/bandwidth/10)
```


## 🤝 Katkıda Bulunma
1.  **Backend** - *Oguzhan Demirbas*
2. **ARAYÜZ(Frontend)** - *Wala Quasem*


