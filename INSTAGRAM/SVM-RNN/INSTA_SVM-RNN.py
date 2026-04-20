"""
Hybrid SVM-RNN Fake Instagram Account Detection System
Uses C-SVM with RBF Kernel for CSV features and Bidirectional LSTM for NLP features
"""

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


class HybridFakeAccountDetector:

    def __init__(self):
        self.svm_model = None
        self.rnn_model = None
        self.scaler = StandardScaler()
        self.tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.max_length = 100
        self.csv_features = None
        self.vocab_size = 5000


    def load_data(self, csv_path='train_csv.csv', json_path='train_nlp.json'):

        print("Loading data...")

        csv_data = pd.read_csv(csv_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        self.csv_features = csv_data.columns.tolist()

        if 'fake' in self.csv_features:
            self.csv_features.remove('fake')

        X_csv = csv_data[self.csv_features].values
        y = csv_data['fake'].values

        X_nlp = []

        for entry in json_data:

            text = f"{entry.get('username','')} {entry.get('fullname','')} {entry.get('bio','')} "
            text += ' '.join(entry.get('captions', []))
            text += ' '.join(entry.get('comments', []))

            X_nlp.append(text)

        print(f"Loaded {len(X_csv)} samples")

        return X_csv, X_nlp, y


    def preprocess_csv(self, X_csv, fit=True):

        if fit:
            X_scaled = self.scaler.fit_transform(X_csv)
        else:
            X_scaled = self.scaler.transform(X_csv)

        return X_scaled


    def preprocess_nlp(self, X_nlp, fit=True):

        if fit:
            self.tokenizer.fit_on_texts(X_nlp)

        sequences = self.tokenizer.texts_to_sequences(X_nlp)

        X_padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )

        return X_padded


    def train_svm(self, X_train, y_train):

        print("\nTraining SVM model...")

        self.svm_model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )

        self.svm_model.fit(X_train, y_train)

        print("SVM training completed!")


    def train_rnn(self, X_train, y_train, X_val, y_val):

        print("\nTraining RNN model...")

        self.rnn_model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        self.rnn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = self.rnn_model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        print("RNN training completed!")

        return history


    def train(self, csv_path='train_csv.csv', json_path='train_nlp.json', test_size=0.2):

        X_csv, X_nlp, y = self.load_data(csv_path, json_path)

        indices = np.arange(len(y))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.1,
            random_state=42,
            stratify=y[train_idx]
        )

        X_csv_train = self.preprocess_csv(X_csv[train_idx], fit=True)
        X_csv_test = self.preprocess_csv(X_csv[test_idx], fit=False)

        X_nlp_list = [X_nlp[i] for i in range(len(X_nlp))]

        X_nlp_train = self.preprocess_nlp([X_nlp_list[i] for i in train_idx], fit=True)
        X_nlp_val = self.preprocess_nlp([X_nlp_list[i] for i in val_idx], fit=False)
        X_nlp_test = self.preprocess_nlp([X_nlp_list[i] for i in test_idx], fit=False)

        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]

        self.train_svm(X_csv_train, y_train)

        self.train_rnn(X_nlp_train, y_train, X_nlp_val, y_val)

        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        self.evaluate(X_csv_test, X_nlp_test, y_test)

        self.save_models()

        return X_csv_test, X_nlp_test, y_test


    def predict(self, X_csv, X_nlp, return_confidence=True):

        svm_proba = self.svm_model.predict_proba(X_csv)[:,1]

        rnn_proba = self.rnn_model.predict(X_nlp, verbose=0).flatten()

        combined_proba = 0.4 * svm_proba + 0.6 * rnn_proba

        predictions = (combined_proba >= 0.5).astype(int)

        if return_confidence:

            confidence = np.where(
                predictions == 1,
                combined_proba * 100,
                (1 - combined_proba) * 100
            )

            return predictions, confidence

        return predictions


    def evaluate(self, X_csv_test, X_nlp_test, y_test):

        predictions, confidence = self.predict(X_csv_test, X_nlp_test)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real  Fake")
        print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")

        print("\nSample Predictions:")
        print("-"*40)

        for i in range(min(10, len(predictions))):

            result = "Fake" if predictions[i] == 1 else "Real"
            actual = "Fake" if y_test[i] == 1 else "Real"
            status = "✓" if predictions[i] == y_test[i] else "✗"

            print(f"{status} {result} ({confidence[i]:.1f}% confidence) - Actual: {actual}")

        # ===== ROC CURVE FOR HYBRID MODEL =====

        svm_proba = self.svm_model.predict_proba(X_csv_test)[:,1]
        rnn_proba = self.rnn_model.predict(X_nlp_test, verbose=0).flatten()

        combined_proba = 0.4 * svm_proba + 0.6 * rnn_proba

        fpr, tpr, thresholds = roc_curve(y_test, combined_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8,6))

        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'Hybrid SVM-RNN (AUC = {roc_auc:.4f})'
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle='--',
            color='gray',
            label='Random Classifier'
        )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve – Fake Instagram Account Detection')
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.show()

        return precision, recall, accuracy


    def save_models(self, svm_path='svm_model.pkl', rnn_path='rnn_model.h5', 
                    scaler_path='scaler.pkl', tokenizer_path='tokenizer.pkl'):

        print("\nSaving models...")

        with open(svm_path, 'wb') as f:
            pickle.dump(self.svm_model, f)

        self.rnn_model.save(rnn_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        print("Models saved successfully!")


# Training Script

if __name__ == "__main__":

    print("="*60)
    print("HYBRID SVM-RNN FAKE ACCOUNT DETECTION SYSTEM")
    print("="*60)

    detector = HybridFakeAccountDetector()

    try:

        detector.train(
            'train_csv.csv',
            'train_nlp.json',
            test_size=0.2
        )

        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:

        print(f"\nError during training: {str(e)}")
        print("Please ensure train_csv.csv and train_nlp.json are in the same directory.")