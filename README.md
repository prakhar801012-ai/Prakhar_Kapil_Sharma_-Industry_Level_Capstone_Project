<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Advanced AutoML Stacking System</title>
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 40px;
        background-color: #0f172a;
        color: #e2e8f0;
        line-height: 1.6;
    }

    h1, h2, h3 {
        color: #38bdf8;
    }

    .container {
        max-width: 1000px;
        margin: auto;
    }

    .card {
        background: #1e293b;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.4);
    }

    code {
        background: #334155;
        padding: 3px 6px;
        border-radius: 5px;
        color: #facc15;
    }

    pre {
        background: #0b1220;
        padding: 15px;
        border-radius: 8px;
        overflow-x: auto;
        color: #22c55e;
    }

    .badge {
        display: inline-block;
        background: #2563eb;
        padding: 5px 10px;
        border-radius: 20px;
        margin-right: 5px;
        font-size: 12px;
    }

    .highlight {
        color: #f97316;
        font-weight: bold;
    }

    ul {
        margin-left: 20px;
    }
</style>
</head>

<body>
<div class="container">

    <h1>🚀 Advanced AutoML Stacking System</h1>
    <p>
        A production-inspired <span class="highlight">AutoML pipeline</span> implementing 
        stacking ensembles, feature selection, drift detection, and threshold optimization.
    </p>

    <div class="card">
        <h2>📌 Key Features</h2>
        <span class="badge">Stacking Ensemble</span>
        <span class="badge">XGBoost</span>
        <span class="badge">LightGBM</span>
        <span class="badge">Random Forest</span>
        <span class="badge">SMOTE</span>
        <span class="badge">Drift Detection</span>
        <span class="badge">Auto Thresholding</span>

        <ul>
            <li>OOF (Out-of-Fold) stacking without data leakage</li>
            <li>Mutual information-based feature selection</li>
            <li>SMOTE handling for imbalanced datasets</li>
            <li>Kolmogorov–Smirnov drift detection</li>
            <li>Optimized classification threshold tuning</li>
        </ul>
    </div>

    <div class="card">
        <h2>🏗️ Architecture</h2>
        <pre>
Input Data
   ↓
Feature Selection (Mutual Information)
   ↓
Scaling (StandardScaler)
   ↓
OOF Cross Validation (5-Fold)
   ↓
Base Models:
   - XGBoost
   - LightGBM
   - Random Forest
   ↓
Meta Model (Logistic Regression)
   ↓
Threshold Optimization
   ↓
Final Prediction
        </pre>
    </div>

    <div class="card">
        <h2>📦 Installation</h2>
        <pre>
pip install scikit-learn xgboost lightgbm imbalanced-learn pandas numpy scipy
        </pre>
    </div>

    <div class="card">
        <h2>🚀 How to Run</h2>
        <pre>
python main.py
        </pre>
        <p>Or in Google Colab:</p>
        <pre>
run()
        </pre>
    </div>

    <div class="card">
        <h2>📊 Sample Output</h2>
        <pre>
🔥 Dataset 1 CV Accuracy: 0.9732
🔥 Dataset 2 CV Accuracy: 0.9874
🔥 Dataset 3 CV Accuracy: 0.9551
🔥 Dataset 4 CV Accuracy: 0.7626

🚀 FINAL SCORE: 0.9196
        </pre>
    </div>

    <div class="card">
        <h2>🧠 System Highlights</h2>
        <ul>
            <li>No data leakage (OOF stacking)</li>
            <li>Adaptive feature selection</li>
            <li>Multi-model ensemble learning</li>
            <li>Robust imbalance handling</li>
            <li>Production-style evaluation pipeline</li>
        </ul>
    </div>

    <div class="card">
        <h2>🚀 Future Upgrades</h2>
        <ul>
            <li>Neural tabular models (FT-Transformer)</li>
            <li>Optuna hyperparameter optimization</li>
            <li>MLflow model tracking</li>
            <li>FastAPI deployment</li>
            <li>Distributed training (Spark / Dask)</li>
        </ul>
    </div>

    <div class="card">
        <h2>🏆 Summary</h2>
        <p>
            This project demonstrates a full-scale <b>AutoML stacking system</b> combining
            classical ML + ensemble learning + optimization techniques.
        </p>
    </div>

</div>
</body>
</html>
