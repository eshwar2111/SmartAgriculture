# Optimal Crop Recommendation System

## Overview
This project provides an **AI-based crop recommendation system** that predicts the most suitable crop for given soil and environmental conditions. The system uses a **Random Forest Classifier** trained on agricultural data collected via a **patented sensor**.

## Features
- Predicts the **optimal crop** based on input parameters such as **NPK levels, temperature, humidity, pH, and rainfall**.
- Uses **Random Forest Classifier** for high-accuracy predictions.
- Processes **real-time sensor data** for dynamic crop recommendations.
- Provides **detailed performance metrics** including accuracy, classification reports, and confusion matrices.

## Dataset
The model is trained on a dataset containing multiple soil and environmental parameters along with the corresponding best-suited crops. The dataset is stored in **`Crop_recommendation.csv`**.

## Installation
Ensure you have **Python 3.x** installed. Install the required dependencies using:
```bash
pip install pandas scikit-learn
```

## Usage
### 1. Training the Model
Run the script to train the Random Forest model:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('/content/drive/MyDrive/Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate model
y_pred = rf_classifier.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
```

### 2. Predicting the Optimal Crop
To predict the best crop based on soil and environmental parameters:
```python
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_df = pd.DataFrame({
        'N': [N], 'P': [P], 'K': [K],
        'temperature': [temperature], 'humidity': [humidity],
        'ph': [ph], 'rainfall': [rainfall]
    })
    predicted_crop = rf_classifier.predict(input_df)
    return predicted_crop[0]

# Example prediction
N, P, K, temp, humidity, ph, rainfall = 57, 8, 38, 30, 45, 6.8, 45
print(f'Optimal Crop: {predict_crop(N, P, K, temp, humidity, ph, rainfall)}')
```

## Model Performance
The model achieves a **high accuracy of 99.32%** on the test dataset. Below is an excerpt from the classification report:
```
              precision    recall  f1-score   support
       rice       1.00      0.89      0.94        19
      mango       1.00      1.00      1.00        19
  watermelon       1.00      1.00      1.00        19
```

## Patent Certificate
This AI system is powered by a **patented sensor** that collects real-time agricultural data.

**Patent Certificate Link:** [(https://drive.google.com/file/d/1OuR2uaMh4eYQXaNLE6w3cjkqUvJbXk0Z/view?usp=sharing)]

## License
This project is licensed under **[MIT License](LICENSE)**.

## Contributors
- **Eshwar B.**
- **Meera R Deepu**



