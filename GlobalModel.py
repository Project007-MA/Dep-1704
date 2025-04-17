import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import optuna
# Load your dataset
df = pd.read_csv('data.csv')
# Assume the target column is named 'target'
X = df.drop('target', axis=1)
y = df['target']
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Reshape for LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
# Train/Test split
X_train_lstm, X_test_lstm, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42)
X_train, X_test = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[2]), X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[2])
def evaluate_model(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print(f"\n{name} Results:")
    print(classification_report(y_true, y_pred))
    print("MCC:", matthews_corrcoef(y_true, y_pred))
    print("Specificity:", specificity)
# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
evaluate_model("Logistic Regression", y_test, lr.predict(X_test))
# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
evaluate_model("SVM", y_test, svm.predict(X_test))
# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
evaluate_model("Random Forest", y_test, rf.predict(X_test))
# BO-LSTM with Optuna
def create_lstm_model(trial):
    model = Sequential()
    model.add(LSTM(units=trial.suggest_int('lstm_units', 32, 64), input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(trial.suggest_float('dropout', 0.2, 0.5)))
    model.add(Dense(1, activation='sigmoid'))
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model
def objective(trial):
    model = create_lstm_model(trial)
    model.fit(X_train_lstm, y_train, epochs=10, batch_size=trial.suggest_int('batch_size', 16, 64), verbose=0)
    y_pred = model.predict(X_test_lstm)
    y_pred = (y_pred > 0.5).astype(int).flatten()
    return matthews_corrcoef(y_test, y_pred)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
best_trial = study.best_trial
model = create_lstm_model(best_trial)
model.fit(X_train_lstm, y_train, epochs=10, batch_size=best_trial.params['batch_size'], verbose=0)
y_pred_lstm = model.predict(X_test_lstm)
y_pred_lstm = (y_pred_lstm > 0.5).astype(int).flatten()
evaluate_model("BO-LSTM", y_test, y_pred_lstm)





