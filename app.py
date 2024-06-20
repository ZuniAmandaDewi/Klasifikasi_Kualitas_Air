import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
from streamlit_option_menu import option_menu

with st.sidebar :
    selected = option_menu ('Klasifikasi',['Introduction', 'Report', 'Implementation', 'Trials'],default_index=0)

 
if (selected == 'Introduction'):
    st.header('Hyperparameter Tuning Menggunakan Grid Search pada Random Forest untuk Klasifikasi Kualitas Air')
    st.markdown('**1. Grid Search**')
    st.write('Parameter merupakan variabel masukan dalam sebuah model, sedangkan hyperparameter adalah variabel yang dapat mempengaruhi hasil model. Dan proses untuk mendapatkan bentuk hyperparameter yang optimal disebut hyperparameter tuning.')
    st.write('Grid Search adalah algoritma ini bekerja dengan mengeksplorasi secara menyeluruh terhadap semua kombinasi hyperparameter yang telah ditentukan pada grid konfigurasi.')
    st.markdown('**2. Random Forest**')
    st.write('Random forest adalah Algoritma ensemble yang digunakan untuk menyelesaikan masalah klasifikasi. Algoritma ensemble merupakan suatu cara untuk meningkatkan keakuratan suatu model klasifikasi dengan menggabungkan algoritma klasifikasi.')
    st.write('Breiman pada tahun 2001 memperkenalkan Algoritma Random Forest yang merupakan gabungan dari beberapa pohon keputusan (Decision Tree). Algoritma Random Forest memiliki beberapa keunggulan, yang diantaranya menimbulkan kesalahan yang relatif kecil, kinerja klasifikasi yang baik, kemampuan memproses data pelatihan dalam jumlah besar secara efisien, dan algoritma dikenal efektif untuk memperkirakan data yang hilang.')

if (selected == 'Report'):
    st.header('Report')
    st.success('Halaman ini berisi grafik dan diagram batang setiap skenario')
    Akurasi, RF, GSRF_S, GSRF_B, GSRF_R = st.tabs(['Akurasi & Waktu', 'RF', 'GSRF_S', 'GSRF_B', 'GSRF_R'])

    with Akurasi:
        st.subheader('Perbandingan Akurasi')
        st.image('akurasi.png')
        st.subheader('Perbandingan Waktu Komputasi')
        st.image('waktu.png')
    
    with RF:
        st.image('rf.png')
        st.image('cm.png')
    
    with GSRF_S:
        st.success('Nilai parameter n-estimators dalam skenario ini dikisaran puluhan')
        st.image('gsrf_s.png')
        st.image('cm_s.png')
    
    with GSRF_B:
        st.success('Nilai parameter n-estimators dalam skenario ini dikisaran ratusan')
        st.image('gsrf_b.png')
        st.image('cm_b.png')
    
    with GSRF_R:
        st.success('Parameter yang digunakan dalam skenario ini berlandaskan pada salah satu jurnal')
        st.image('gsrf_r.png')
        st.image('cm_r.png')

if (selected == 'Implementation'):
    st.title("Implementasi Data")
    
    st.markdown('**N_estimators**')
    jumlah_kolom_tree = st.number_input("Masukkan jumlah parameter n_estimators yang ingin di uji:", min_value=1, step=1, value=1)
    hasil_input_tree = []
    # Tampilkan kolom input sesuai dengan jumlah yang diminta
    for i in range(jumlah_kolom_tree):
        input_value = st.number_input(f"Kolom input n_estimators {i+1}", step=1)
        hasil_input_tree.append(input_value)

    st.markdown('**Max_depth**')
    jumlah_kolom_depth = st.number_input("Masukkan jumlah parameter max_depth yang ingin di uji:", min_value=1, step=1, value=1)
    hasil_input_depth = []
    # Tampilkan kolom input sesuai dengan jumlah yang diminta
    for i in range(jumlah_kolom_depth):
        input_value = st.number_input(f"Kolom input max_depth {i+1}", step=1)
        hasil_input_depth.append(input_value)
    
    st.markdown('**Min_samples_split**')
    jumlah_kolom_split = st.number_input("Masukkan jumlah parameter min_samples_split yang ingin di uji:", min_value=1, step=1, value=1)
    hasil_input_split = []
    # Tampilkan kolom input sesuai dengan jumlah yang diminta
    for i in range(jumlah_kolom_split):
        input_value = st.number_input(f"Kolom input min_samples_split {i+1}", 2,step=1)
        hasil_input_split.append(input_value)
    
    st.markdown('**Max_features**')
    jumlah_kolom_feature = st.number_input("Masukkan jumlah parameter max_fetures yang ingin di uji:", min_value=1, step=1, value=1)
    hasil_input_feature = []
    # Tampilkan kolom input sesuai dengan jumlah yang diminta
    for i in range(jumlah_kolom_feature):
        input_value = st.number_input(f"Kolom input max_features {i+1}", step=1)
        hasil_input_feature.append(input_value)

    st.markdown('**K-Fold Cross Validation**')
    jumlah_fold = st.number_input("Masukkan jumlah fold :", 2, step=1)
    
    # Tombol untuk menampilkan hasil input
    if st.button("Tampilkan Hasil Input"):
        missing_value = ['#NUM!', np.nan]
        data=pd.read_csv("waterQuality1.csv", na_values = missing_value)

        # Menghapus missing values
        missing = data.isnull().sum()
        kolom_tidak_null = missing[missing != 0].index.tolist()
        data.dropna(subset=kolom_tidak_null, axis=0, inplace=True)
        df = data.reset_index(drop=True)
        # st.dataframe(df)

        #Split data
        X = pd.read_csv('dataX.csv')  #data fitur
        y = df.is_safe #data class
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        #Grid Search & RF
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': hasil_input_tree,
            'max_depth': hasil_input_depth,
            'min_samples_split' : hasil_input_split,
            'max_features': hasil_input_feature
        }
        grid_search_rf = GridSearchCV(estimator=model, param_grid=param_grid, cv=jumlah_fold, n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        best_params_rf = grid_search_rf.best_params_
        st.write('**Parameter terbaik pada Random Forest**')
        best_params_rf
        rf = RandomForestClassifier(**best_params_rf)
        rf.fit(X_train, y_train)
        akurasi_rf = rf.score(X_test, y_test)
        st.write('**Akurasi Random Forest**', akurasi_rf)


if (selected == "Trials"):
    st.title('Uji Coba')
    chloramine = st.number_input('Masukkan Nilai Chloramine', placeholder=0.0, format="%0.5f")
    chromium = st.number_input('Masukkan Nilai Chromium', placeholder=0.0, format="%0.5f")
    bacteria = st.number_input('Masukkan Nilai Bacteria', placeholder=0.0, format="%0.5f")
    viruses = st.number_input('Masukkan Nilai Viruses', placeholder=0.0, format="%0.5f")
    perchlorate = st.number_input('Masukkan Nilai Perchlorate', placeholder=0.0, format="%0.5f")
    silver = st.number_input('Masukkan Nilai Silver', placeholder=0.0, format="%0.5f")
    submit = st.button("Submit")
    if submit :
            inputs = np.array([chloramine, chromium, bacteria, viruses, perchlorate, 
                    silver]).reshape(1,-1)
            print(inputs)
        

            gsrf_b=joblib.load('gsrf_b.joblib')

        
            hasil_rf=gsrf_b.predict(inputs)
            # Menampilkan pesan berdasarkan hasil prediksi
            if hasil_rf == 0.0 :
                st.success("Berdasarkan algoritma Random Forest, maka data tersebut dinyatakan : **Tidak Aman**")
            else:
                st.success("Berdasarkan algoritma Random Forest, maka data tersebut dinyatakan : **Aman**")
