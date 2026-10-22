import pandas as pd
import joblib

model=joblib.load("customer_model.pkl")

new_customer = pd.DataFrame([{
    'age': 29,
    'tenure_months': 4,
    'monthly_charges': 95,
    'contract_type': 0,
    'payment_method': 0,
    'num_support_calls': 5,
    'internet_service': 2,
    'num_services': 2,
    'late_payments': 3
}])

label = model.predict(new_customer)[0]
prob = model.predict_proba(new_customer)[0, 1]

print('Prediction :','WILL LEAVE' if label == 1 else 'WILL STAY')