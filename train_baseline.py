import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Загружаем наши очищенные данные
df = pd.read_csv('credit_cleaned.csv')

# Encoding: превращаем текст в числа
# Колонки типа 'RENT', 'OWN' станут отдельными столбцами с 0 и 1
df_encoded = pd.get_dummies(df)

# Разделяем на признаки (X) и цель (y)
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

# Сплит данных 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучаем бейзлайн (Логистическая регрессия)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оценка
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

print(f"Baseline ROC-AUC: {auc:.4f}")
print("\nОтчет по классификации:")
print(classification_report(y_test, model.predict(X_test)))

# Сохраняем модель и список колонок
joblib.dump(model, 'baseline_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')