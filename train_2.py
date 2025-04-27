import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
# สมมติว่ามี DataFrame ชื่อ df
# ตัวอย่างสร้าง data เอง
df = pd.read_csv('Housing.csv')
X,y = df.drop(columns=['price']), df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# สร้าง model
model = LinearRegression()

# เทรน model
model.fit(X_train, y_train)

# ทำนาย
y_pred = model.predict(X_test)

# ประเมินผล
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')

# ดูค่าสมการ
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

with open("app/model_custom.pkl", "wb") as f:
    pickle.dump(model, f)
