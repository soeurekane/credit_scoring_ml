import pandas as pd

try:
    df = pd.read_csv('credit_risk_dataset.csv')
    print("Файл успешно загружен")
    
    print("\n--- Пропуски в данных ---")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\n--- Статистика по возрасту и доходу ---")
    print(df[['person_age', 'person_income', 'loan_amnt']].describe())

except FileNotFoundError:
    print("Файл 'credit_risk_dataset.csv' не найден в папке")