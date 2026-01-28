import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('credit_cleaned.csv')
df_encoded = pd.get_dummies(df)

X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Обучаем XGBoost
print("Обучаем XGBoost...")
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)

# Загружаем Baseline для сравнения
lr_model = joblib.load('baseline_model.pkl')

# 4. Считаем метрики
lr_probs = lr_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, lr_probs)
xgb_auc = roc_auc_score(y_test, xgb_probs)

print(f"\nРезультаты:")
print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(f"XGBoost AUC: {xgb_auc:.4f}")

# 5. Визуализация Feature Importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Топ-10 важных факторов для выдачи кредита')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\n📈 График важности признаков сохранен как feature_importance.png")

# Сохраняем лучшую модель
joblib.dump(xgb_model, 'best_credit_model.pkl')
# Сохраняем названия колонок
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("Модель XGBoost и колонки сохранены!")