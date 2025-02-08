# Telco Company Churn Prediction

**Article Report** : [Medium](https://medium.com/@febbyngrni/predicting-customer-churn-an-end-to-end-ml-for-telco-company-fb62d3d83e56)<br>

IndiHouse adalah perusahaan telekomunikasi yang menyediakan layanan telepon, internet, dan paket hiburan streaming. Perusahaan menghadapi tantangan dalam mempertahankan pelanggan di tengah persaingan yang ketat. Tingginya churn pelanggan berdampak negatif pada pendapatan dan meningkatkan biaya akuisisi pelanggan baru.

Untuk mengatasi masalah ini, IndiHouse ingin menerapkan prediksi churn berbasis machine learning. Dengan model ini, perusahaan dapat mengidentifikasi pelanggan berisiko churn lebih awal dan mengambil langkah intervensi, seperti penawaran khusus atau peningkatan kualitas layanan.

### Objectives
- Membangun model prediktif untuk mengklasifikasikan pelanggan yang kemungkinan akan churn.
- Menggunakan model tersebut untuk strategi preventif dalam menurunkan churn rate.
- Mengintegrasikan model dengan aplikasi Streamlit untuk akses prediksi real-time.

### Modeling Task
Prediksi churn dilakukan menggunakan algoritma **Logistic Regression, KNN, Decision Tree, Random Forest, dan XGBoost**. Model dievaluasi dengan **Accuracy dan F1 Score** untuk memastikan kinerjanya dalam mengidentifikasi pelanggan berisiko churn.


## Preprocessing
1. **One-Hot Encoding**: Mengubah categorical features menjadi format numerik dan menyimpan hasil encoding sebagai file .pkl untuk digunakan kembali.
2. **Label Balancing**: Menyeimbangkan data churn dengan tiga metode:
    - Undersampling: Mengurangi sampel kelas mayoritas.
    - Oversampling: Menambah sampel kelas minoritas.
    - SMOTE: Membuat sampel sintetis pada kelas minoritas.
3. **Label Encoding**: Mengonversi target variable (churn) ke format numerik setelah balancing, lalu menyimpan data hasil balancing sesuai dengan metode yang digunakan.


## Modeling
Modeling dilakukan dengan beberapa algoritma, seperti Logistic Regression, KNN, Decision Tree, Random Forest, dan XGBoost.
| Model                 | Configuration  | F1-Score      |
|-----------------------|----------------|---------------|
| **XGBoost**           | **Oversampling** | **0.717229** |
| Random Forest         | Oversampling   | 0.715829      |
| Logistic Regression   | Oversampling   | 0.714059      |
| Logistic Regression   | Undersampling  | 0.709756	     |
| XGBoost               | SMOTE          | 0.706947	     |
| etc..         	    |                |               |

XGBoost Oversampling memiliki F1-Score tertinggi sebesar 0.71 dan akurasi sebesar 0.76, menunjukkan bahwa model ini mampu menangkap pola dari data dengan baik dan memberikan prediksi yang paling akurat dibandingkan model lainnya.

Setelah dilakukan hyperparameter tuning, F1-Score meningkat menjadi 0.81, menunjukkan bahwa optimasi parameter mampu meningkatkan performa model dalam mengklasifikasikan pelanggan yang berpotensi churn.

### Evaluation Test Set
``` 
    Classification Report Test Set
    
                  precision    recall  f1-score   support
    
               1       0.51      0.72      0.60       281
               0       0.88      0.75      0.81       776
    
        accuracy                           0.74      1057
       macro avg       0.69      0.73      0.70      1057
    weighted avg       0.78      0.74      0.75      1057
```

- Prediksi Churn:
    - Model memprediksi churn dengan precision yang cukup rendah (0.51), menunjukkan adanya false positive yang cukup tinggi.
    - Recall churn cukup baik (0.72), berarti model mampu menangkap sebagian besar pelanggan yang benar-benar churn.
- Prediksi Non-Churn:
    - Model memiliki performa lebih baik dengan F1-Score sebesar 0.81.
    - Terdapat false negative yang cukup banyak, artinya ada pelanggan yang sebenarnya churn tetapi diprediksi sebagai non-churn.
- Akurasi:
    - Akurasi pada test set sebesar 0.74, mengalami sedikit penurunan dibandingkan setelah hyperparameter tuning.
    - Akurasi ini masih sebanding dengan model baseline, menunjukkan bahwa tuning tidak memberikan peningkatan akurasi yang signifikan.
 
## Implementation
Untuk menjalankan model ini pada streamlit gunakan command ini:
``` bash
  streamlit run app.py
```
![streamlit](https://github.com/user-attachments/assets/343ff5ce-cb92-4064-afea-9703d2d1fe1c)

## Future Work
- Threshold Tuning: Menyesuaikan ambang batas prediksi untuk meningkatkan recall tanpa terlalu banyak mengorbankan precision.
- Fitur Tambahan: Memanfaatkan data perilaku pelanggan dan interaksi layanan untuk meningkatkan akurasi prediksi.
- Eksplorasi Model Lain: Mencoba algoritma seperti LightGBM atau CatBoost yang mungkin lebih optimal dalam menangani dataset ini.
- Hyperparameter Tuning Lanjutan: Menggunakan Grid Search atau Random Search untuk menemukan parameter terbaik yang dapat lebih meningkatkan performa model.
