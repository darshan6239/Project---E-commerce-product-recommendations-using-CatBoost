import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#  Step 2: Load the Dataset
import pandas as pd

df = pd.read_excel('Online-Retail.xlsx')
df.head()

#  Step 3: Data Preprocessing
df.dropna(inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df = df[df['Quantity'] > 0]

df.head()

#  Step 4: Feature Engineering
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique', 
    'TotalPrice': 'sum' 
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.head()



#  Step 5: Train-Test Split
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = np.where(rfm['Monetary'] > rfm['Monetary'].median(), 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


#  Step 6: Train the CatBoost Model
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=False)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')


#  Step 7: Visualize Feature Importance
feature_importance = model.get_feature_importance()
features = X.columns

importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()


#    Step 8: Interactive GUI with ipywidgets
import ipywidgets as widgets
from IPython.display import display

recency_input = widgets.IntSlider(min=0, max=365, step=1, description='Recency')
frequency_input = widgets.IntSlider(min=0, max=100, step=1, description='Frequency')
monetary_input = widgets.FloatSlider(min=0, max=10000, step=0.01, description='Monetary')

output = widgets.Output()

def make_prediction(recency, frequency, monetary):
    data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
    prediction = model.predict(data)[0]
    return "Recommend" if prediction == 1 else "Do not recommend"

def update_prediction(change):
    with output:
        output.clear_output()
        prediction = make_prediction(recency_input.value, frequency_input.value, monetary_input.value)
        print(f'Recommendation: {prediction}')

recency_input.observe(update_prediction, names='value')
frequency_input.observe(update_prediction, names='value')
monetary_input.observe(update_prediction, names='value')

display(recency_input, frequency_input, monetary_input, output)
