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
| Random Forest         | Undersampling  | 0.704584	     |
| Logistic Regression   | SMOTE          | 0.702416      |
| Random Forest         | SMOTE          | 0.702151      |
| XGBoost               | Undersampling  | 0.700168      |
| K-Nearest Neighbors   | Undersampling  | 0.665075      |
| Decision Tree         | Undersampling  | 0.660932      |	
| K-Nearest Neighbors   | SMOTE          | 0.654981      |	
| Decision Tree         | SMOTE          | 0.652504      |	
| K-Nearest Neighbors   | Oversampling   | 0.647383      |	
| Decision Tree         | Oversampling   | 0.641981      |

XGBoost Oversampling memiliki F1-Score tertinggi sebesar 0.71 dan akurasi sebesar 0.76, menunjukkan bahwa model ini mampu menangkap pola dari data dengan baik dan memberikan prediksi yang paling akurat dibandingkan model lainnya.
