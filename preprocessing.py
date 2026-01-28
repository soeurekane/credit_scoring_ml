import pandas as pd

def clean_data(df):
    # Удаляем явные аномалии
    df = df[df['person_age'] < 100]
    
    # Заполняем пропуски в стаже (person_emp_length) медианой
    emp_median = df['person_emp_length'].median()
    df['person_emp_length'] = df['person_emp_length'].fillna(emp_median)
    
    # Заполняем пропуски в ставке (loan_int_rate) медианой
    int_rate_median = df['loan_int_rate'].median()
    df['loan_int_rate'] = df['loan_int_rate'].fillna(int_rate_median)
    
    print("Очистка завершена: аномалии удалены, пропуски заполнены.")
    return df

if __name__ == "__main__":
    df = pd.read_csv('credit_risk_dataset.csv')
    df_clean = clean_data(df)
    
    # Проверяем результат
    print(f"Новый макс возраст: {df_clean['person_age'].max()}")
    print(f"Осталось пропусков: {df_clean.isnull().sum().sum()}")
    
    # Сохраняем очищенные данные
    df_clean.to_csv('credit_cleaned.csv', index=False)