"""
Hybrid Balanced Random Forest + Bi-LSTM RNN Fake Facebook Account Detection System
Uses Balanced Random Forest for CSV features and Bidirectional LSTM for NLP features

Adapted for Facebook profiles with conditional feature selection:
  - Personal accounts (personal(0)/page(1) == 0):
      Uses all numerical features EXCEPT: #followers, #following, category
  - Page accounts (personal(0)/page(1) == 1):
      Uses all numerical features EXCEPT: private, #friends, friends visibility
"""

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Feature definitions for Facebook accounts
# ---------------------------------------------------------------------------
ALL_CSV_FEATURES = [
    'profile pic', 'cover pic', 'fullname characters', 'nums/character fullname',
    'bio length', 'external url', 'private', 'personal(0)/page(1)',
    '#friends', '#followers', '#following', 'friends visibility',
    'category', 'workplace', 'education'
]

# Features excluded per account type
PERSONAL_EXCLUDE = {'#followers', '#following', 'category'}
PAGE_EXCLUDE     = {'private', '#friends', 'friends visibility'}

# Common features always used (excluding 'personal(0)/page(1)' which is only a
# selector flag, not a predictive feature, and 'fake' which is the label)
ACCOUNT_TYPE_COL = 'personal(0)/page(1)'

def get_features_for_type(account_type: int) -> list:
    """Return the feature list appropriate for the given account type (0 or 1)."""
    exclude = PERSONAL_EXCLUDE if account_type == 0 else PAGE_EXCLUDE
    return [f for f in ALL_CSV_FEATURES
            if f != ACCOUNT_TYPE_COL and f not in exclude]


# ---------------------------------------------------------------------------
# Helper: build a padded feature matrix for a mixed-type dataset
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame, personal_cols: list, page_cols: list) -> np.ndarray:
    """
    Construct a feature matrix where each row uses the columns appropriate to
    that account's type.  Missing columns for a given type are filled with 0.

    The union of personal_cols and page_cols defines the final column order.
    """
    all_cols = sorted(set(personal_cols) | set(page_cols))  # deterministic order
    result = np.zeros((len(df), len(all_cols)), dtype=np.float64)

    col_index = {c: i for i, c in enumerate(all_cols)}

    for row_i, (_, row) in enumerate(df.iterrows()):
        account_type = int(row[ACCOUNT_TYPE_COL])
        cols = personal_cols if account_type == 0 else page_cols
        for col in cols:
            result[row_i, col_index[col]] = row[col]

    return result, all_cols


class HybridBRFRNNDetector:
    def __init__(self):
        self.brf_model      = None
        self.rnn_model      = None
        self.scaler         = StandardScaler()
        self.tokenizer      = Tokenizer(num_words=5000, oov_token='<OOV>')
        self.max_length     = 100
        self.vocab_size     = 5000
        self.feature_cols   = None   # final ordered list of CSV feature columns
        self.personal_cols  = None
        self.page_cols      = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, csv_path='CSV.csv', json_path='NLP.json'):
        """Load CSV and JSON data for Facebook accounts."""
        print("Loading data...")

        # ---------- CSV ----------
        csv_data = pd.read_csv(csv_path)
        # Drop fully-empty rows (trailing blank rows in the file)
        csv_data.dropna(how='all', inplace=True)
        csv_data.reset_index(drop=True, inplace=True)
        # Fill remaining NaNs with 0
        csv_data.fillna(0, inplace=True)

        # Derive per-type feature lists
        self.personal_cols = get_features_for_type(0)
        self.page_cols     = get_features_for_type(1)

        # Build mixed feature matrix
        X_csv, self.feature_cols = build_feature_matrix(
            csv_data, self.personal_cols, self.page_cols
        )
        y = csv_data['fake'].values.astype(int)

        # ---------- JSON ----------
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        X_nlp = []
        for entry in json_data:
            # Combine ALL available textual fields for Facebook
            text  = f"{entry.get('fullname', '')} "
            text += f"{entry.get('bio', '')} "
            text += f"{entry.get('work', '')} "
            text += f"{entry.get('education', '')} "
            text += f"{entry.get('categories', '')} "
            text += ' '.join(entry.get('captions',  []))
            text += ' '.join(entry.get('comments',  []))
            X_nlp.append(text.strip())

        print(f"Loaded {len(X_csv)} samples")
        print(f"Personal account features ({len(self.personal_cols)}): {self.personal_cols}")
        print(f"Page account features     ({len(self.page_cols)}):     {self.page_cols}")
        print(f"Combined feature matrix shape: {X_csv.shape}")
        return X_csv, X_nlp, y

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def preprocess_csv(self, X_csv, fit=True):
        """Scale numerical features."""
        if fit:
            return self.scaler.fit_transform(X_csv)
        return self.scaler.transform(X_csv)

    def preprocess_nlp(self, X_nlp, fit=True):
        """Tokenise and pad text sequences."""
        if fit:
            self.tokenizer.fit_on_texts(X_nlp)
        sequences = self.tokenizer.texts_to_sequences(X_nlp)
        return pad_sequences(sequences, maxlen=self.max_length,
                             padding='post', truncating='post')

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def train_brf(self, X_train, y_train):
        """Train Balanced Random Forest."""
        print("\nTraining Balanced Random Forest model...")
        self.brf_model = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            sampling_strategy='auto',
            replacement=True
        )
        self.brf_model.fit(X_train, y_train)
        print("Balanced Random Forest training completed!")

    def train_rnn(self, X_train, y_train, X_val, y_val):
        """Train Bidirectional LSTM RNN."""
        print("\nTraining Bi-LSTM RNN model...")

        self.rnn_model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=128,
                      input_length=self.max_length),
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
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )

        print("Bi-LSTM RNN training completed!")
        return history

    # ------------------------------------------------------------------
    # Full training pipeline
    # ------------------------------------------------------------------
    def train(self, csv_path='CSV.csv', json_path='NLP.json', test_size=0.2):
        """Complete training pipeline."""
        X_csv, X_nlp, y = self.load_data(csv_path, json_path)

        # 80/20 train-test split
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=y
        )
        # Further split training into train/validation
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.1, random_state=42, stratify=y[train_idx]
        )

        # CSV features
        X_csv_train = self.preprocess_csv(X_csv[train_idx], fit=True)
        X_csv_val   = self.preprocess_csv(X_csv[val_idx],   fit=False)
        X_csv_test  = self.preprocess_csv(X_csv[test_idx],  fit=False)

        # NLP features
        X_nlp_train = self.preprocess_nlp([X_nlp[i] for i in train_idx], fit=True)
        X_nlp_val   = self.preprocess_nlp([X_nlp[i] for i in val_idx],   fit=False)
        X_nlp_test  = self.preprocess_nlp([X_nlp[i] for i in test_idx],  fit=False)

        y_train = y[train_idx]
        y_val   = y[val_idx]
        y_test  = y[test_idx]

        # Train
        self.train_brf(X_csv_train, y_train)
        self.train_rnn(X_nlp_train, y_train, X_nlp_val, y_val)

        # Evaluate
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        self.evaluate(X_csv_test, X_nlp_test, y_test)

        self.save_models()
        return X_csv_test, X_nlp_test, y_test

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X_csv, X_nlp, return_confidence=True):
        """Hybrid prediction: BRF (40%) + RNN (60%)."""
        brf_proba = self.brf_model.predict_proba(X_csv)[:, 1]
        rnn_proba = self.rnn_model.predict(X_nlp, verbose=0).flatten()

        combined_proba = 0.4 * brf_proba + 0.6 * rnn_proba
        predictions    = (combined_proba >= 0.5).astype(int)

        if return_confidence:
            confidence = np.where(
                predictions == 1,
                combined_proba * 100,
                (1 - combined_proba) * 100
            )
            return predictions, confidence

        return predictions

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X_csv_test, X_nlp_test, y_test):
        """Evaluate and print performance metrics."""
        predictions, confidence = self.predict(X_csv_test, X_nlp_test)

        precision = precision_score(y_test, predictions)
        recall    = recall_score(y_test, predictions)
        accuracy  = accuracy_score(y_test, predictions)
        cm        = confusion_matrix(y_test, predictions)

        print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")

        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real  Fake")
        print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")

        print("\nSample Predictions:")
        print("-" * 40)
        for i in range(min(10, len(predictions))):
            result = "Fake" if predictions[i] == 1 else "Real"
            actual = "Fake" if y_test[i]     == 1 else "Real"
            status = "✓" if predictions[i] == y_test[i] else "✗"
            print(f"{status} {result} ({confidence[i]:.1f}% confidence) - Actual: {actual}")

        return precision, recall, accuracy

    # ------------------------------------------------------------------
    # Single-account prediction
    # ------------------------------------------------------------------
    def predict_single(self, csv_row: dict, text_data: str):
        """
        Predict for a single Facebook account.

        Parameters
        ----------
        csv_row   : dict  – all raw CSV fields for the account (including
                            'personal(0)/page(1)' so the correct feature set
                            is selected automatically).
        text_data : str   – concatenated text content for the account.

        Returns
        -------
        result     : str   – "Fake" or "Real"
        confidence : float – confidence percentage
        """
        account_type = int(csv_row.get(ACCOUNT_TYPE_COL, 0))
        cols = self.personal_cols if account_type == 0 else self.page_cols

        # Build feature vector aligned to the trained feature column order
        col_index  = {c: i for i, c in enumerate(self.feature_cols)}
        csv_vector = np.zeros(len(self.feature_cols))
        for col in cols:
            if col in col_index:
                csv_vector[col_index[col]] = float(csv_row.get(col, 0))

        csv_scaled   = self.scaler.transform(csv_vector.reshape(1, -1))
        nlp_processed = self.preprocess_nlp([text_data], fit=False)

        prediction, confidence = self.predict(csv_scaled, nlp_processed)
        result = "Fake" if prediction[0] == 1 else "Real"
        return result, confidence[0]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_models(self, brf_path='fb_brf_model.pkl', rnn_path='fb_rnn_model.h5',
                    scaler_path='fb_scaler.pkl', tokenizer_path='fb_tokenizer.pkl',
                    meta_path='fb_meta.pkl'):
        """Save all model artefacts."""
        print("\nSaving models...")

        with open(brf_path, 'wb') as f:
            pickle.dump(self.brf_model, f)

        self.rnn_model.save(rnn_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)

        # Save feature column metadata
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'feature_cols':  self.feature_cols,
                'personal_cols': self.personal_cols,
                'page_cols':     self.page_cols
            }, f)

        print("Models saved successfully!")

    def load_models(self, brf_path='fb_brf_model.pkl', rnn_path='fb_rnn_model.h5',
                    scaler_path='fb_scaler.pkl', tokenizer_path='fb_tokenizer.pkl',
                    meta_path='fb_meta.pkl'):
        """Load all model artefacts."""
        print("Loading models...")

        with open(brf_path, 'rb') as f:
            self.brf_model = pickle.load(f)

        self.rnn_model = load_model(rnn_path)

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
            self.feature_cols  = meta['feature_cols']
            self.personal_cols = meta['personal_cols']
            self.page_cols     = meta['page_cols']

        print("Models loaded successfully!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("HYBRID BALANCED RANDOM FOREST + BI-LSTM RNN")
    print("FAKE FACEBOOK ACCOUNT DETECTION SYSTEM")
    print("=" * 60)

    detector = HybridBRFRNNDetector()

    try:
        detector.train('CSV.csv', 'NLP.json', test_size=0.2)
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please ensure CSV.csv and NLP.json are in the same directory.")