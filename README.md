# Laporan Proyek Machine Learning - Andrew Benedictus Jamesie

## Daftar Isi

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preprocessing](#data-preprocessing)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Kesimpulan](#kesimpulan)
- [Referensi](#referensi)

## Project Overview

Dalam proyek ini akan membahas permasalahan mengenai minat membaca buku masyarakat Indonesia yang berhubungan langsung dengan perkembangan teknologi seperti media sosial dan *e-commerce* di Indonesia.

<img src="https://user-images.githubusercontent.com/64983961/189553721-da2a9077-a686-4788-9329-2e514b2e9168.png" alt="E-Book Illustration" title="E-Book Illustration" width="100%">

Dalam beberapa dekade terakhir, perkembangan teknologi dan juga internet sangat mempengaruhi minat membaca buku masyarakat Indonesia. Contohnya adalah hadirnya media sosial, dan platform seperti YouTube, Netflix, Amazon, dan layanan web sejenis yang lainnya membuat masyarakat lebih banyak menyukai menggunakan platform tersebut sebagai hiburan dan sumber untuk memperoleh informasi. Berbeda pada zaman dahulu di mana masyarakat Indonesia masih bergantung pada media cetak contohnya adalah buku untuk memperoleh ilmu dan informasi.

Membaca sebagai salah satu dari empat keterampilan dalam berbahasa, merupakan suatu proses yang dilakukan serta digunakan oleh pembaca untuk memperoleh pesan yang ingin disampaikan oleh penulis melalui media kata-kata atau bahasa tulis. [[1]](https://core.ac.uk/display/287170379?utm_source=pdf&utm_medium=banner&utm_campaign=pdf-decoration-v1 'Permasalahan Budaya Membaca di Indonesia (Studi Pustaka Tentang Problematika & Solusinya)') Selain itu, membaca juga memerlukan sebuah kemampuan untuk memahami apa yang dibaca sehingga dapat menemukan pesan implisit, dan dapat mengaplikasikan pengetahuan yang terdapat dalam teks melalui proses sintesis dari berbagai gagasan dan informasi. [[2]](#referensi 'Pengajaran Membaca di Sekolah Dasar')

Menurut data yang diambil dari picodi.com, terjadi peningkatan paling tinggi terhadap permintaan jumlah buku di toko buku *online* pada bulan Desember, yaitu sebesar 12% dari semua transaksi tahunan. Sedangkan penurunan yang paling signifikan terjadi pada bulan Juni sebesar 6% dari semua transaksi tahunan. Berdasarkan survei tentang bagaimana cara pelanggan yang membeli buku, sebanyak 47% membeli buku di toko buku konvensional, sekitar 37% meminjamnya dari perpustakaan, dan hanya 12% yang meminjam dari teman. Sedangkan 10% menyatakan bahwa tidak banyak membaca atau tidak tertarik pada buku. Mengingat semakin populernya teknologi dan internet, ternyata sebesar 55% responden lebih suka memesan buku secara *online*, sedangkan yang lebih memilih untuk membeli buku di toko buku konvensional sebesar 73%. [[3]](https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia 'Pembelian Buku di Indonesia (dan di seluruh Dunia)')

Salah satu situs media sosial yang populer untuk membaca dan menuliskan cerita, yaitu [Wattpad](https://www.wattpad.com 'Wattpad - Where stories live'). Wattpad merupakan *platform* untuk membuat komunitas membaca dan menghapus penghalang antara pembaca dan penulis. Menurut opini dari Restu I. Aji, dilansir dari Quora.com, beberapa karya yang terdapat pada Wattpad bahkan lebih berwarna dari pada buku yang terbit secara konvensional atau cetak. Menurutnya, tidak sedikit buku yang populer di Wattpad yang justru dialih-mediakan ke buku fisik dan hadir di toko buku *offline* disertai embel-embel "telah dibaca sekian ratus ribu/juta kali di Wattpad". [[4]](https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline 'Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?')

Dari permasalahan dan latar belakang di atas, maka di dalam proyek ini akan dibuat sebuah model *machine learning* berupa *recommendation system* atau sistem rekomendasi untuk menentukan rekomendasi buku yang terbaik kepada pengguna. Model ini nantinya dapat digunakan dan di-*deploy* untuk keperluan tertentu, misalnya diterapkan di dalam katalog buku, daftar buku perpustakaan, media sosial seperti Wattpad, ataupun pada *e-commerce* yang menjual buku baik digital maupun cetak.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data buku, pengguna, dan *rating* atau penilaian agar dapat digunakan sebagai informasi untuk membuat model *machine learning* sistem rekomendasi?
2. Bagaimana cara membuat model *machine learning* untuk sistem rekomendasi buku?

### Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data sehingga data siap digunakan pada model *machine learning* untuk sistem rekomendasi.
2. Membuat model *machine learning* untuk sistem rekomendasi buku terbaik kepada pengguna.

### Solution Statements

Di bawah ini merupakan gambar diagram alir kerja (*workflow*) yang dilakukan dalam mengerjakan proyek ini.

<img src="https://user-images.githubusercontent.com/64983961/189554698-c07b9c98-180b-46cd-9e3c-c7898d3e4da4.png" alt="Recommendation System Workflow" title="Recommendation System Workflow">

Berdasarkan tujuan dari proyek yang telah dipaparkan di atas, maka berikut adalah beberapa solusi yang dapat dilakukan agar dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap pra-pemrosesan data atau *data preprocessing* merupakan tahap untuk mengubah data mentah atau *raw data* menjadi data yang bersih atau *clean data* yang siap untuk digunakan pada proses selanjutnya. Tahap ini dapat dilakukan dengan cara, yaitu:
    - Melakukan penyesuaian dan mengubah nama kolom atau atribut sehingga memudahkan ketika proses pemanggilan *dataset* beserta nama atribut atau kolom tertentu.
    - Menggabungkan data yang terpisah sehingga dapat digunakan pada tahap selanjutnya.
    
2. Tahap persiapan data atau *data preparation* merupakan proses transformasi pada data sehingga data menjadi bentuk yang cocok untuk melakukan proses pemodelan di tahap selanjutnya. Tahap ini dapat dilakukan dengan beberapa teknik, yaitu:
    - Melakukan pengecekan nilai data yang kosong, tidak ada, ataupun *null* (*missing value*) dan menghapus data tersebut atau mengganti/mengisinya dengan suatu nilai tertentu.
    - Melakukan pengecekan data yang mungkin duplikat agar tidak akan mengganggu hasil dari pemodelan dan sistem yang telah dibangun.
    
3. Tahap pembuatan model *machine learning* untuk sistem rekomendasi buku adalah jenis algoritma sistem rekomendasi yang terpersonalisasi atau *personalized recommender system*. Pembuatan model akan menggunakan dua (2) pendekatan, yaitu *content-based filtering recommendation*, dan pendekatan *collaborative filtering recommendation*.
    - **Content-based Filtering Recommendation**  
      Sistem rekomendasi yang berbasis konten (*content-based filtering*) merupakan sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan item yang disukai oleh pengguna di masa lalu. *Content-based filtering* akan mempelajari profil minat pengguna baru berdasarkan data dari objek yang telah dinilai oleh pengguna lain sebelumnya. Pada pendekatan menggunakan *content-based filtering* akan menggunakan algoritma TF-IDF Vectorizer dan Cosine Similarity.
      - TF-IDF Vectorizer  
        Algoritma Term Frequency Inverse Document Frequency Vectorizer (TF-IDF Vectorizer) adalah algoritma yang dapat melakukan kalkulasi dan transformasi dari teks mentah menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks serta dapat digunakan dan dimengerti oleh model *machine learning*. [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 'TfidfVectorizer - sci-kit Documentation')
        
        Kelebihan dari teknik ini adalah tidak membutuhkan data yang diperoleh dari pengguna lain karena rekomendasi yang akan diberikan akan spesifik hanya untuk pengguna tersebut. Sedangkan kekurangan dengan menggunakan teknik ini ialah hasil rekomendasi yang hanya terbatas dari pengguna itu saja dan tidak dapat memperluas data dari penilaian pengguna lain. TF-IDF dapat dihitung menggunakan rumus sebagai berikut: [[6]](https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530 'TF-IDF Simplified - Towards Data Science')
        
        $$idf_i=log \left( \frac{n}{df_i} \right)$$
        
        Di mana $idf_i$ merupakan skor IDF untuk *term* $i$; $df_i$ adalah jumlah dokumen yang mengandung *term* $i$; dan $n$ adalah jumlah total dokumen. Semakin tinggi nilai $df$ suatu *term*, maka semakin rendah $idf$ untuk *term* tersebut. Ketika jumlah $df$ sama dengan $n$ yang berarti istilah/*term* tersebut muncul di semua dokumen, $idf$ akan menjadi 0, karena $log(1)=0$.
        
        Sedangkan nilai TF-IDF merupakan perkalian dari matriks frekuensi *term* dengan IDF-nya.
        
        $$w_{i,j}=tf_{i,j} \times idf_i$$
        
        Di mana $w_{i,j}$ merupakan skor TF-IDF untuk *term* $i$ pada dokumen $j$; $tf_{i,j}$ adalah frekuensi *term* untuk *term* $i$ pada dokumen $j$, dan $idf_i$ adalah skor $idf$ untuk *term* $i$.
        
      - Cosine Similarity  
        Teknik *cosine similarity* digunakan untuk melakukan perhitungan derajat kesamaan (*similarity degree*) antara dua sampel. [[7]](https://www.sciencedirect.com/topics/computer-science/cosine-similarity 'Cosine Similarity - ScienceDirect Topics')
        
        $$S_c(A,B)=cos(\theta)= \frac{A \times B}{\|A\| \|B\|} = \frac{\displaystyle\sum^{n}_{i=1} A_iB_i}{\sqrt{\displaystyle\sum^{n}_{i=1} A^{2}_{i} } \sqrt{\displaystyle\sum^{n}_{i=1} B^{2}_{i}} }$$
        
        Di mana $A_i$ dan $B_i$ merupakan komponen dari masing-masing vektor A dan B.
        
    - **Collaborative Filtering Recommendation**  
      Sistem rekomendasi yang berbasis penyaringan kolaboratif (*collaborative filtering*) adalah sistem rekomendasi yang memberikan rekomendasi item yang hampir sama dengan preferensi pengguna di masa lalu berdasarkan riwayat pengguna lain yang memiliki preferensi yang sama, misalnya berdasarkan penilaian atau *rating* yang telah diberikan pengguna di masa lalu. [[8]](https://realpython.com/build-recommendation-engine-collaborative-filtering 'Build a Recommendation Engine With Collaborative Filtering - Real Python') Namun, teknik ini memilki kekurangan yaitu, tidak dapat memberikan rekomendasi item yang tidak memiliki riwayat penilaian/*rating* atau transaksi.
      
      Menggunakan teknik *collaborative filtering recommendation* akan memerlukan proses penyandian (*encoding*) fitur-fitur yang terdapat pada *dataset* ke dalam bentuk indeks integer, lalu memetakannya ke dalam *dataframe* yang berkaitan. Kemudian akan dilakukan pembagian distribusi **dataset** dengan rasio tertentu untuk memisahkan data latih (*training data*) dan juga data uji (*validation data*) sebelum dilakukan tahap pemodelan.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Data Understanding

[<img src="https://user-images.githubusercontent.com/64983961/189554742-1d36ac13-035a-4130-8722-92b37420e416.png" alt="Book Recommendation Kaggle Dataset" title="Book Recommendation Kaggle Dataset" width="100%">](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset. Di bawah ini adalah informasi detail tentang *dataset* yang digunakan.

|                         | Keterangan                                                                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset: Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset 'Build state-of-the-art models for book recommendation system') |
| *Usability*             | 10.00                                                                                                                                                                              |
| Lisensi                 | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0 'Creative Common - CC0 1.0 Universal')                                                                      |
| Penilaian/*Rating*      | Silver                                                                                                                                                                             |
| Jenis dan Ukuran Berkas | zip (25 MB)                                                                                                                                                                        |
| Kategori                | Literature, Art, Culture and Humanities                                                                                                                                            |

Dalam dataset tersebut berisi tiga (3) berkas CSV ([Comma-separated Values](https://en.wikipedia.org/wiki/Comma-separated_values 'Wikipedia - Comma-separated values')), yaitu `Books.csv`, `Ratings.csv`, `Users.csv`.

- **Books.csv**, memiliki atribut atau fitur sebagai berikut,
  
  <img src="https://user-images.githubusercontent.com/64983961/189554870-865889f3-55e8-4ca7-8d57-58aabbf3ad06.png" alt="Deskripsi Variabel Books" title="Deskripsi Variabel Books">
  
  - `ISBN` : *International Standard Book Number*
  - `Book-Title` : Judul buku
  - `Book-Author` : Penulis buku
  - `Year-Of-Publication` : Tahun terbit buku
  - `Publisher` : Penerbit buku
  - `Image-URL-S` : Tautan sampul buku ukuran kecil
  - `Image-URL-M` : Tautan sampul buku ukuran sedang
  - `Image-URL-L` : Tautan sampul buku ukuran besar
  
- **Ratings.csv**, memiliki atribut atau fitur sebagai berikut,

  <img src="https://user-images.githubusercontent.com/64983961/189554969-86a96dd9-365f-40f8-83b9-28a8115349e9.png" alt="Deskripsi Variabel Ratings" title="Deskripsi Variabel Ratings">
  
  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `ISBN` : *International Standard Book Number*
  - `Book-Rating` : *Rating* buku yang diberikan pengguna
  
- **Users.csv**, memiliki atribut atau fitur sebagai berikut,
  
  <img src="https://user-images.githubusercontent.com/64983961/189555005-eb7094f5-9875-4dfb-baa6-b6dcaeda11ff.png" alt="Deskripsi Variabel Users" title="Deskripsi Variabel Users">
  
  - `User-ID` : Identitas unik pengguna berupa bilangan bulat atau integer
  - `Location` : Lokasi tempat tinggal pengguna
  - `Age` : Umur pengguna

Deskripsi statistik untuk *dataset* `ratings` pada fitur `Book-Rating` dapat dilihat pada gambar di bawah ini.

<img src="https://user-images.githubusercontent.com/64983961/189554983-d73a1d05-509f-431d-8fbf-650beae2e541.png" alt="Deskripsi Statistik Ratings" title="Deskripsi Statistik Ratings">

Dari gambar di atas dapat dilihat bahwa terdapat,
- Total jumlah data (`count`) sebanyak 1.149.780;
- Rata-rata *rating* (`mean`) 3;
- Simpangan baku/standar deviasi *rating* (`std`) 4;
- *Rating* Minimal (`min`), kuartil bawah/Q1 *rating* (`25%`), kuartil tengah/Q2/median *rating* (`50%`) 0;
- Kuartil atas/Q3 *rating* (`75%`) 7;
- *Rating* maksimum (`max`) 10

Berikut adalah visualisasi grafik histogram frekuensi sebaran data *rating* pengguna terhadap buku yang sudah pernah dibaca, mulai dari *rating* 1 hingga *rating* 10.

<img src="https://user-images.githubusercontent.com/64983961/189555048-8885fa77-3d34-4abc-a793-972589637fd1.png" alt="Grafik Histogram Frekuensi Sebaran Data Rating" title="Grafik Histogram Frekuensi Sebaran Data Rating" width="100%">

Berdasarkan hasil visualisasi grafik histogram "Jumlah Rating Buku" di atas, dapat disimpulkan bahwa *rating* terbanyak dari buku yang sudah pernah dibaca adalah *rating* 0, dengan jumlah *rating* kira-kira sebanyak lebih dari 700.000. *Rating* 0 tersebut dapat menyebabkan bias dan mempengaruhi hasil analisis, sehingga data dengan *rating* 0 tersebut dapat dihapus pada tahap [*data preparation*](#data-preparation).

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Data Preprocessing

Pada tahap pra-pemrosesan data atau *data preprocessing* dilakukan untuk mengubah data mentah (*raw data*) menjadi data yang bersih (*clean data*) yang siap untuk digunakan pada proses selanjutnya. Ada beberapa tahap yang dilakukan pada *data preprocessing*, yaitu:
- **Mengubah Nama Kolom/Atribut/Fitur**  
  Proses pengubahan nama kolom atau atribut atau fitur dari masing-masing *dataframe* bertujuan untuk memudahkan proses pemanggilan *dataframe* tersebut.
  - Books
    
    <img src="https://user-images.githubusercontent.com/64983961/189573236-955e8cf5-bca2-4b61-9383-9bca264aaecc.png" alt="Preprocessing Data Books" title="Preprocessing Data Books">
    
  - Ratings
    
    <img src="https://user-images.githubusercontent.com/64983961/189555153-8a4ce0c1-0a96-4d1e-a95b-3d92c455242c.png" alt="Preprocessing Data Ratings" title="Preprocessing Data Ratings">
    
  - Users
    
    <img src="https://user-images.githubusercontent.com/64983961/189555165-4ccdb1b4-2c10-47ad-8541-6f480bc803f7.png" alt="Preprocessing Data Users" title="Preprocessing Data Users">
    
- **Menggabungkan Data ISBN**  
  Penggabungan data ISBN buku dilakukan menggunakan fungsi `.concatenate` dengan bantuan *library*  [`numpy`](https://numpy.org). Data ISBN terdapat pada *dataframe* buku dan *dataframe*  *rating*, sehingga dilakukan penggabungan data tersebut pada atribut atau kolom `isbn`.
  
- **Menggabungkan Data User**  
  Penggabungan data `user_id` buku dilakukan menggunakan fungsi `.concatenate` dengan bantuan *library*  [`numpy`](https://numpy.org). Data `user_id` terdapat pada *dataframe*  *rating* dan *dataframe*  *user*, sehingga dilakukan penggabungan data tersebut pada atribut atau kolom `user_id`.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan nantinya. Ada beberapa tahap yang dilakukan pada *data preparation*, yaitu:
- **Pengecekan *Missing Value***  
  Proses pengecekan data yang kosong, hilang, *null*, atau *missing value* ditemukan pada *dataframe* `books`, sehingga data yang *missing* tersebut dihapus atau di-*drop*.
  
  Sedangkan pada *dataframe* `ratings` tidak ditemukan adanya *missing value*, tetapi perlu dilakukan penghapusan *rating* 0 dikarenakan *rating* 0 tersebut merupakan rating yang paling banyak berdasarkan proses [*data understanding*](#data-understanding) sebelumnya, yaitu sebanyak 716.109 data dengan *rating* 0. Hal tersebut dapat menyebabkan bias pada hasil analisis data. Kategori *rating* 0 tidak diikutsertakan, sehingga diperoleh hasil visualisasi grafik histogram di bawah ini.
  
  <img src="https://user-images.githubusercontent.com/64983961/189555311-bf883e83-363a-49dd-93f5-abf384e734db.png" alt="Grafik Rating Setelah Perbaikan Missing Value" title="Grafik Rating Setelah Perbaikan Missing Value" width="100%">
  
  Berdasarkan hasil visualisasi grafik histogram di atas dengan *rating* 0 yang telah dihapus, dapat dilihat distribusi frekuensi data yang lebih rapi dan jelas, terutama pada data *rating* 1 hingga *rating* 4.
  
  Kemudian pada *dataframe* `Users`, terdapat sebanyak 110.762 *missing value* pada fitur umur. Sehingga data tersebut dapat diganti atau diisi dengan nilai modus atau nilai yang paling sering muncul dalam data umur tersebut.
  
  <img src="https://user-images.githubusercontent.com/64983961/189555376-3a3bd63e-4556-4c67-8f64-00d7a1e2c309.png" alt="Grafik Umur Users Setelah Perbaikan Missing Value" title="Grafik Umur Users Setelah Perbaikan Missing Value">
  
  Berdasarkan hasil visualisasi grafik histogram umur pengguna di atas dapat dilihat bahwa rentang umur pengguna paling banyak berada pada umur 20 hingga 30-an.
  
- **Pengecekan Data Duplikat**  
  Melakukan pengecekan data duplikat atau data yang sama pada masing-masing *dataframe*. Hasilnya tidak ada data yang duplikat dari ketiga *dataframe*.
  
- **Data Buku dan Rating**  
  Melakukan penggabungan atau *merge* data buku dan *rating* menjadi sebuah *dataframe*.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Modeling

Tahap selanjutnya adalah proses _modeling_ atau membuat model _machine learning_ yang dapat digunakan sebagai sistem rekomendasi untuk menentukan rekomendasi buku yang terbaik kepada pengguna dengan beberapa algoritma sistem rekomendasi tertentu.

Berdasarkan tahap pemahaman data atau [*data understanding*](#data-understanding 'Data Understanding') sebelumnya, dapat dilihat bahwa data untuk masing-masing _dataframe_, yaitu data buku, _rating_, dan _users_ tergolong data yang cukup banyak, mencapai ratusan hingga jutaan data. Hal tersebut akan berdampak pada biaya yang akan diperlukan untuk melakukan proses pemodelan _machine learning_, seperti memakan waktu yang lama dan _resource_ RAM ataupun GPU yang cukup besar. Oleh karena itu, dalam kasus ini data yang akan digunakan untuk proses pemodelan _machine learning_ data akan dibatasi hanya 10.000 baris data buku dan 5000 baris data _rating_.

```python
books   = books[:10000]
ratings = ratings[:5000]
```

1. **Content-based Recommendation**
   - TF-IDF Vectorizer  
     TF-IDF Vectorizer akan mentransformasikan teks menjadi representasi angka yang memiliki makna tertentu dalam bentuk matriks. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 5.575 data *author* atau penulis buku.
     
     <img src="https://user-images.githubusercontent.com/64983961/189569829-bc2722db-c89b-4d8c-89fb-a4ec784beee3.png" alt="Tabel Hasil TF-IDF Vectorizer" title="Tabel Hasil TF-IDF Vectorizer">

   - *Cosine Similarity*  
     *Cosine Similarity* akan melakukan perhitungan derajat kesamaan (*similarity degree*) antar judul buku. Ukuran matriks yang diperoleh adalah sebesar 10.000 data buku dan 10.000 data buku juga.
     
     <img src="https://user-images.githubusercontent.com/64983961/189569832-eaddebce-e7b1-45a7-a666-10a9469067b5.png" alt="Tabel Hasil TF-IDF Vectorizer" title="Tabel Hasil TF-IDF Vectorizer">
     
   - Hasil *Top-N Recommendation*
     Hasil pengujian sistem rekomendasi dengan pendekatan *content-based recommendation* adalah sebagai berikut.
     
     <img src="https://user-images.githubusercontent.com/64983961/189569194-f9db3e41-7429-4ac4-b486-ff1c178098a6.png" alt="Content Based Pilih Buku" title="Content Based Pilih Buku" width="100%">
     
     Pada gambar di atas merupakan data berdasarkan judul buku yang dipilih oleh pengguna.
     
     <img src="https://user-images.githubusercontent.com/64983961/189569197-edcf7f3e-1f42-4bdf-a464-5a87566110f6.png" alt="Content Based Hasil" title="Content Based Hasil">
     
     Dapat dilihat bahwa sistem yang telah dibangun berhasil memberikan rekomendasi beberapa judul buku berdasarkan input atau masukan sebuah judul buku, yaitu "Proxies", dan diperoleh beberapa judul buku yang berdasarkan perhitungan sistem.
     
2. **Collaborative Filtering Recommendation**
   - Data Preparation  
     *Data preparation* yang dilakukan adalah dengan melakukan penyandian (*encoding*) fitur `user_id` dan `isbn` pada *dataframe* `ratings` ke dalam bentuk indeks integer. Kemudian melakukan pemetaan fitur yang telah di-*encoding* tersebut ke dalam masing-masing *dataframe* yang `ratings`.
     
     Diperoleh jumlah *user* sebesar 1204, jumlah buku sebesar 4565, nilai minimal *rating* yaitu 1, dan nilai maksimum *rating* yaitu 10.
     
   - Split Training Data dan Validation Data  
     Tahap ini dilakukan pengacakan *dataframe ratings* terlebih dahulu, lalu kemudian membagi data dengan rasio 80:20, di mana 80% untuk data latih (*training data*) dan 20% sisanya adalah untuk data uji (*validation data*).
     
     <img src="https://user-images.githubusercontent.com/64983961/189555657-9262d0ad-75cc-4d07-85da-c6b05172bdb6.png" alt="Train Test Split Fraction" title="Train Test Split Fraction">
   
   - Model Development dan Hasil  
     Berdasarkan model yang telah di-*training*, berikut adalah hasil pengujian sistem rekomendasi buku dengan pendekatan *collaborative filtering recommendation*.
     
     <img src="https://user-images.githubusercontent.com/64983961/189569257-e1665749-d50e-4af4-92b1-fe1c6351a071.png" alt="Collaborative Filtering Hasil" title="Collaborative Filtering Hasil">
     
     Berdasarkan hasil di atas, dapat dilihat bahwa sistem akan mengambil pengguna secara acak, yaitu pengguna dengan `user_id`  **885**. Lalu akan dicari buku dengan *rating* terbaik dari user tersebut, yaitu,
     * **The Joy Luck Club** oleh **Amy Tan**
     * **Awakening** oleh **Kate Chopin**
     
     Kemudian sistem akan membandingkan antara buku dengan *rating* tertinggi dari *user* dan semua buku, kecuali buku yang telah dibaca tersebut, lalu akan mengurutkan buku yang akan direkomendasikan berdasarkan nilai rekomendasi yang tertinggi. Dapat dilihat terdapat 10 daftar buku yang direkomendasikan oleh sistem.
     
     Dapat dibandingkan antara ***Book with high ratings from user*** dan ***Top 10 Books Recommendation***, terdapat buku dengan penulis atau *author* yang sama, yaitu **The Kitchen God's Wife** oleh **Amy Tan**. Dengan begitu, dapat dikatakan bahwa sistem yang telah dibangun dapat merekomendasikan buku kepada pengguna dengan prediksi yang cukup sesuai.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Evaluation

1. **Content-based Recommendation**  
   Pada tahap evaluasi untuk model sistem rekomendasi dengan pendekatan *content-based recommendation* dapat menggunakan evaluasi dengan metrik akurasi yang diperoleh dari,
   
   $$Accuracy=\frac{\displaystyle\sum_{i=1}^{n} RecommendedBooks_i}{\displaystyle\sum_{i=1}^{n} BooksWithSameAuthor_i} \times 100$$
   
   Masih menggunakan data yang sama pada tahap [Modeling](#modeling 'Modeling') *content-based recommendation*, pada proses Hasil *Top-N Recommendation*, yaitu penulis buku atau `book_author` Toni Morrison, akan dilakuakn proses pencarian jumlah judul buku atau `book_title` dengan penulis atau *author* yang sama. Pencarian tersebut menggunakan variabel baru yang di mana akan mengambil sebuah data buku yang telah dibaca oleh pengguna. Hasil yang diperoleh adalah Toni Morrison memiliki jumlah buku sebanyak 7 buah buku.
   
   ```python
   accuracy = (recommendations.shape[0] / readedBookSameAuthor) * 100
   accuracy
   ```
   
   Proses perhitungan akurasi dilakukan dengan membagi banyaknya rekomendasi buku yang dihasilkan, dibagi dengan banyaknya jumlah buku yang ditulis oleh *author* atau penulis yang sama, kemudian dikalikan dengan 100. Sehingga diperoleh nilai **akurasi** sebesar **57.14%**.
   
2. **Collaborative Filtering Recommendation**  
   Berdasarkan model *machine learning* yang sudah dibangun menggunakan *embedding layer* dengan *Adam optimizer* dan *binary crossentropy loss function*, metrik yang digunakan adalah *Root Mean Squared Error* (RMSE). Perhitungan RMSE dapat dilakukan menggunakan rumus berikut,
   
   $$RMSE=\sqrt{\sum^{n}_{i=1} \frac{y_i - y\\_pred_i}{n}}$$
   
   Di mana, nilai $n$ merupakan jumlah *dataset*, nilai $y_i$ adalah nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya terdahap $i$ sebagai urutan data dalam *dataset*.
   
   Hasil nilai RMSE yang rendah menunjukkan bahwa variasi nilai yang dihasilkan dari model sistem rekomendasi mendekati variasi nilai observasinya. Artinya, semakin kecil nilai RMSE, maka akan semakin dekat nilai yang diprediksi dan diamati.
   
   Berikut merupakan visualisasi hasil *training* dan *validation error* dari metrik RMSE serta *training* dan *validation loss* ke dalam grafik plot.
   
   <img src="https://user-images.githubusercontent.com/64983961/189569295-2e5881ff-1038-46f4-8598-9af7df7291b0.png" alt="Model Training Plot" title="Model Training Plot">

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Kesimpulan

Kesimpulannya adalah model yang digunakan untuk melakukan rekomendasi buku berdasarkan teknik *Content-based Recommendation* dan teknik *Collaborative Filtering Recommendation* telah berhasil dibuat dan sesuai dengan preferensi pengguna. Pada *collaborative filtering* diperlukan data *rating* dari pengguna, sedangkan pada *content-based filtering*, data *rating* tidak diperlukan karena analisis sistem rekomendasi akan berdasarkan atribut item dari masing-masing buku. Sistem rekomendasi dengan pendekatan *content-based recommendation* dan *collaborative filtering recommendation* memiliki kelebihan dan kekurangannya masing-masing.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')

## Referensi
[1] L. Tahmidaten and W. Krismanto, "Permasalahan Budaya Membaca di Indonesia (Studi Pustaka Tentang Problematika & Solusinya)", _Scholaria: Jurnal Pedidikan dan Kebudayaan_, vol. 10, no, 1, pp. 22-23, Jan. 2020, doi: 10.24246/j.js.2020.v10.i1.p22-33, Retrieved from: https://ejournal.uksw.edu/scholaria/article/view/2656.

[2] F. Rahim, _Pengajaran Membaca di Sekolah Dasar_, Jakarta: Sinar Grafika, 2008.

[3] Picodi, "Pembelian Buku di Indonesia (dan di seluruh Dunia)", _Picodi.com_, 2019, Retrieved from: https://www.picodi.com/id/mencari-penawaran/pembelian-buku-di-indonesia-dan-di-seluruh-dunia.

[4] "Apakah kamu lebih suka baca buku di situs Wattpad atau toko buku offline?", *Quora*, Retrieved from: https://id.quora.com/Apakah-kamu-lebih-suka-baca-buku-di-situs-Wattpad-atau-toko-buku-offline.

[5] scikit-learn, "sklearn.feature_extraction.text.TfidfVectorizer", Retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.

[6] L. Ramadhan, "TF-IDF Simplified, A short introduction to TF-IDF vectorizer", *Towards Data Science*, 2021, Retrieved from: https://towardsdatascience.com/tf-idf-simplified-aba19d5f5530.

[7] ScienceDirect, "Cosine Similarity", Retrieved from: https://www.sciencedirect.com/topics/computer-science/cosine-similarity.

[8] A. Ajitsaria, "Build a Recommendation Engine With Collaborative Filtering", *Real Python*, 2019, Retrieved from: https://realpython.com/build-recommendation-engine-collaborative-filtering.

[← Kembali ke Daftar Isi](#daftar-isi 'Daftar Isi')
