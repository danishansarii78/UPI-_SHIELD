# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Script started...")

# 1. Load the dataset
print("Loading data...")
df = pd.read_csv('creditcard.csv')

# 2. Prepare the data
# The dataset is imbalanced (very few frauds). This is typical.
# 'Class' is the target variable: 1 for fraud, 0 for normal.
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Data prepared...")

# 3. Train the XGBoost model
# The `scale_pos_weight` is important for imbalanced datasets.
# It tells the model to pay more attention to the rare fraud cases.
fraud_count = y_train.value_counts()[1]
non_fraud_count = y_train.value_counts()[0]
scale = non_fraud_count / fraud_count

print(f"Scale Pos Weight: {scale:.2f}")

print("Training model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale)
model.fit(X_train, y_train)

print("Model training complete.")

# 4. Evaluate the model
print("Evaluating model...")
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print("\nClassification Report:\n")
# Look at the 'precision' and 'recall' for class 1 (fraud). These are better metrics.
print(classification_report(y_test, preds))

# 5. Save the trained model
# We'll save the model to a file so our API can use it later.
joblib.dump(model, 'fraud_model.pkl')
print("Model saved as fraud_model.pkl")

print("Script finished.")