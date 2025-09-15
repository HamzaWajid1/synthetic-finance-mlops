# 🚀 Synthetic Finance Fraud & Anomaly Detection with MLOps

This project focuses on **detecting anomalies and fraudulent activities** in synthetic financial transaction data.  
It combines **machine learning, statistical inference, and MLOps best practices** to create a production-ready pipeline.

Developed as part of **UC Santa Cruz – Intro to Machine Learning**,  
but extended into a **portfolio-grade project** to showcase skills for AI/ML roles.

---

## ✨ Features
- ✅ Data cleaning & feature engineering on synthetic finance dataset  
- ✅ Anomaly detection (Isolation Forest, LOF, Autoencoder)  
- ✅ Fraud classification (XGBoost, Random Forest)  
- ✅ Explainability (SHAP values, statistical analysis)  
- ✅ MLOps practices: MLflow tracking, Prefect/Airflow pipelines, monitoring  
- ✅ REST API with FastAPI for fraud detection  
- ✅ Containerization with Docker (Kubernetes optional)  
- ✅ Big Data support via PySpark & Dask  
- ✅ Interactive Streamlit dashboard for anomaly insights  

---

## 🛠 Tech Stack
- **Machine Learning**: Scikit-learn, XGBoost, PyTorch (Autoencoder), SHAP  
- **MLOps**: MLflow, Prefect/Airflow, DVC (optional)  
- **API**: FastAPI, Uvicorn  
- **Deployment**: Docker, Kubernetes  
- **Big Data**: PySpark, Dask  
- **Visualization**: Matplotlib, Seaborn, Plotly, Streamlit  

---

## ⚡ Installation
Clone repo & create environment:  

```bash
git clone https://github.com/HamzaWajid1/synthetic-finance-mlops.git
cd synthetic-finance-mlops

# Create venv
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1️⃣ Preprocess Data
```bash
python src/preprocess.py
```

### 2️⃣ Train Model
```bash
python src/train.py
```

### 3️⃣ Run FastAPI Service
```bash
uvicorn api.main:app --reload
```

**Endpoint examples:**
- `POST /predict` → Input transaction JSON → Returns anomaly score/fraud label
- `GET /health` → Health check

### 4️⃣ Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## 📊 Results

- **Models compared:** Isolation Forest, LOF, Autoencoder, XGBoost
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations:** anomaly clusters, SHAP importance, fraud detection over time

## 🛳 Deployment

- **Local:** Dockerized API + dashboard
- **Cloud (optional):** Kubernetes manifest included in `docker/k8s-deployment.yaml`

## 🔮 Future Work

- Real-time fraud detection with Kafka streams
- Bayesian inference for uncertainty estimation
- Extension to cryptocurrency fraud detection datasets

## 📜 License

MIT License – free to use, modify, and distribute.

## 🙌 Acknowledgments

- **Dataset:** TestDataBox Synthetic Finance Dataset
- **Inspiration:** Real-world fraud detection systems in fintech & banking
