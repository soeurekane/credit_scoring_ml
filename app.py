from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Загружаем модель и колонки
model = joblib.load('best_credit_model.pkl')
model_columns = joblib.load('model_columns.pkl')

app = FastAPI(title="Credit Scoring API")

# Описываем структуру входных данных
class LoanApplication(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str  
    person_emp_length: float
    loan_intent: str           
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.get("/")
def home():
    return {"message": "Система кредитного скоринга активна"}

@app.post("/predict")
def predict(app: LoanApplication):
    # Превращаем входящие данные в DataFrame
    data = pd.DataFrame([app.dict()])
    
    data_encoded = pd.get_dummies(data)
    
    # Добавляем недостающие колонки
    final_df = pd.DataFrame(columns=model_columns)
    final_df = pd.concat([final_df, data_encoded]).fillna(0)
    final_df = final_df[model_columns]
    
    # Делаем предсказание
    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]
    
    status = "Одобрено" if prediction == 0 else "Отказ (высокий риск)"
    
    return {
        "status": status,
        "probability_of_default": round(float(probability), 2)
    }