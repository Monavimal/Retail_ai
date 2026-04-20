# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from textblob import TextBlob
from flask import Flask, request, jsonify



#======================================
# 2. LOAD DATASET (Updated for Render)
# =========================================
# This finds the file inside your GitHub folder instead of your C: drive
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'retail_data.csv')

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Dataset not found! Please ensure retail_data.csv is in your GitHub repo.")


# =========================================
# 3. DATA CLEANING
# =========================================
# Drop missing values
df.dropna(inplace=True)

# Convert date column
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Feature Engineering
df['TotalRevenue'] = df['Quantity'] * df['Price']

# Create Churn Flag (if last purchase > 90 days)
latest_date = df['OrderDate'].max()
df['DaysSinceLastPurchase'] = (latest_date - df['OrderDate']).dt.days
df['Churn'] = df['DaysSinceLastPurchase'].apply(lambda x: 1 if x > 90 else 0)

print("\nCleaned Data:")
print(df.head())

# =========================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================

# Sales over time
df.groupby('OrderDate')['TotalRevenue'].sum().plot(figsize=(10,5), title="Sales Trend")
plt.show()

# Top products
top_products = df.groupby('Product')['TotalRevenue'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='bar', title="Top Products")
plt.show()

# =========================================
# 5. CHURN PREDICTION MODEL
# =========================================

# Aggregate customer-level data
customer_df = df.groupby('CustomerID').agg({
    'TotalRevenue': 'sum',
    'Quantity': 'sum',
    'DaysSinceLastPurchase': 'min',
    'Churn': 'max'
}).reset_index()

# Features & Target
X = customer_df[['TotalRevenue', 'Quantity', 'DaysSinceLastPurchase']]
y = customer_df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================================
# 6. SENTIMENT ANALYSIS (AI FEATURE)
# =========================================

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Apply sentiment analysis
if 'Review' in df.columns:
    df['SentimentScore'] = df['Review'].apply(get_sentiment)
    print("\nSentiment Analysis Sample:")
    print(df[['Review', 'SentimentScore']].head())

# =========================================
# 7. SIMPLE RECOMMENDATION SYSTEM
# =========================================

# Create product co-occurrence matrix
basket = df.groupby(['CustomerID', 'Product'])['Quantity'].sum().unstack().fillna(0)

# Correlation
corr_matrix = basket.corr()

def recommend(product_name):
    if product_name in corr_matrix:
        return corr_matrix[product_name].sort_values(ascending=False).head(5)
    else:
        return "Product not found"

print("\nRecommendations for a sample product:")
print(recommend(df['Product'].iloc[0]))

#=========================================
# 8. FLASK API (Updated for Render)
# =========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "Retail AI API Running"



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    input_data = np.array([[
        data['TotalRevenue'],
        data['Quantity'],
        data['DaysSinceLastPurchase']
    ]])
    
    prediction = model.predict(input_data)[0]
    
    return jsonify({
        "Churn Prediction": int(prediction)
    })


 



if __name__ == "__main__":
    # Render assigns a dynamic port, so we must use os.environ.get
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)