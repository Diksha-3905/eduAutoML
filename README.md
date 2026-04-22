# 🎓 eduAutoML

**A beginner-friendly, explainable AutoML system — built for students, by a student.**

![GitHub stars](https://img.shields.io/github/stars/Diksha-3905/eduAutoML?style=social)
![GitHub forks](https://img.shields.io/github/forks/Diksha-3905/eduAutoML?style=social)
![GitHub license](https://img.shields.io/github/license/Diksha-3905/eduAutoML)
[![PyPI version](https://badge.fury.io/py/eduAutoML.svg)](https://pypi.org/project/eduAutoML/)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

> **Stop copy-pasting ML code you don't understand.**  
> eduAutoML trains models, explains predictions, teaches you *why* each decision was made — all in one tool.

---

## ✨ What's New in v0.2

| Feature | Description |
|---|---|
| 🔬 **SHAP Explainability** | See *why* your model makes each prediction |
| 📊 **Auto EDA** | Missing values, correlation heatmaps, distributions — one call |
| 🧠 **Learning Mode** | Explains classification, overfitting, scaling in plain English |
| 🏆 **Model Comparison** | Visual leaderboard of all trained models |
| ⚡ **XGBoost + LightGBM** | Industry-standard boosting models included |
| 🎨 **Gradio Dashboard** | Drag-and-drop, 5-tab UI — no code needed |
| 💾 **Save & Load** | Export your trained model as `.pkl` |

---

## 🚀 Quick Start

### Install

```bash
pip install eduautoml
# or from source:
git clone https://github.com/Diksha-3905/eduAutoML && cd eduAutoML
pip install -e .
```

### 3 Lines to Train

```python
import pandas as pd
from eduautoml import AutoML

df = pd.read_csv("your_data.csv")

automl = AutoML(learning_mode=True)   # ← flip this on to learn as you go
automl.fit(df, target="target_column")
```

**Sample output:**
```
10:23:01 [INFO] 🔍 Detecting problem type...
10:23:01 [INFO]    → Task detected: CLASSIFICATION
10:23:01 [INFO] LEARNING MODE | 📚 Classification predicts a CATEGORY...
10:23:01 [INFO] ⚙️  Preprocessing data...
10:23:01 [INFO] 🏋️  Training 6 models...
10:23:01 [INFO]    Training LogisticRegression...
10:23:02 [INFO]    ✅ LogisticRegression → 0.8667  (0.1s)
10:23:02 [INFO]    Training RandomForestClassifier...
10:23:03 [INFO]    ✅ RandomForestClassifier → 0.9467  (1.2s)
10:23:05 [INFO]    ✅ XGBClassifier → 0.9600  (1.8s)
10:23:05 [INFO] 🏆 Best model: XGBClassifier (score=0.9600)
10:23:05 [INFO] LEARNING MODE | ⚡ XGBoost took the crown — industry-standard boosting...
```

---

## 📊 Auto EDA

```python
stats, chart_paths = automl.eda_report(df, target="Species")
```

Generates:
- 📉 Missing value bar chart
- 🎯 Target class distribution
- 🔥 Correlation heatmap
- 📈 All feature distributions

---

## 🔬 Explainability

```python
automl.explain()
```

Generates:
- **Feature Importance** chart — which features matter most
- **SHAP Summary** — direction & magnitude of each feature's impact
- **Confusion Matrix** — where the model gets confused

```
10:25:12 [INFO] 📊 Feature importance chart saved → feature_importance.png
10:25:13 [INFO] 🔬 Computing SHAP values...
10:25:15 [INFO] 🔬 SHAP summary saved → shap_summary.png
10:25:15 [INFO] 📊 Confusion matrix saved → confusion_matrix.png
```

---

## 🏆 Model Comparison

```python
comparison_df, chart_path = automl.compare_models()
print(comparison_df)
```

| Model | Accuracy | Time (s) |
|---|---|---|
| XGBClassifier | 0.9600 | 1.8 |
| RandomForestClassifier | 0.9467 | 1.2 |
| GradientBoostingClassifier | 0.9200 | 0.9 |
| LogisticRegression | 0.8667 | 0.1 |

---

## 💾 Save & Load

```python
# Save
automl.save("my_model.pkl")

# Load later
from eduautoml import AutoML
automl = AutoML.load("my_model.pkl")
predictions = automl.predict(new_data)
```

---

## 🎨 Gradio Dashboard (No-Code UI)

```bash
python gradio_app.py
```

Opens a 5-tab web app:

1. **📂 Upload** — drag-and-drop CSV, pick target column
2. **📊 EDA** — instant visual analysis
3. **🏋️ Train** — run all models, see leaderboard
4. **🔬 Explain** — SHAP + feature importance charts
5. **💾 Save** — download `.pkl` model

---

## 🧠 Learning Mode

Turn on `learning_mode=True` to get plain-English explanations at each step:

- *"Why was Random Forest chosen?"*
- *"What is overfitting? (Detected: Train=0.98, Test=0.71 ⚠️)"*
- *"What does StandardScaler do?"*
- *"What is a confusion matrix?"*

Perfect for students who want to understand ML, not just get outputs.

---

## 🏗️ Architecture

```
eduautoml/
├── automl.py          # Main AutoML class (fit, explain, eda_report, save/load)
├── __init__.py        # Public API
└── core/
    ├── eda.py         # EDA utilities
    └── ...

gradio_app.py          # 5-tab Gradio dashboard
```

**Models trained by default:**
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost *(if installed)*
- LightGBM *(if installed)*

---

## 🛠 Roadmap

- ✅ Auto task detection (classification / regression)
- ✅ Auto preprocessing (imputation, encoding, scaling)
- ✅ Multi-model training + leaderboard
- ✅ SHAP explainability
- ✅ Auto EDA report
- ✅ Learning Mode
- ✅ Gradio 5-tab dashboard
- ✅ Save & load models
- 🔜 Auto report PDF export
- 🔜 Hyperparameter tuning (Optuna)
- 🔜 One-click FastAPI deployment

---

## 🤝 Contributing

```bash
git clone https://github.com/Diksha-3905/eduAutoML
cd eduAutoML
pip install -e ".[dev]"
git checkout -b feature/your-idea
# make changes → PR
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 📚 Who Is This For?

| Audience | Use Case |
|---|---|
| 🎓 ML Students | Learn how AutoML works step-by-step |
| 👩‍💻 Educators | Demo ML pipelines in class |
| 🧑‍💼 Hackathon teams | Quick baseline ML in minutes |
| 🔍 Curious learners | Peek under the AutoML hood |

---

## ⭐ Star This Project

If eduAutoML helped you learn or saved you time — **give it a star** 🌟  
It helps other students find this tool and keeps the community growing.

---

*Built with ❤️ by [Diksha Wagh](https://github.com/Diksha-3905)*
