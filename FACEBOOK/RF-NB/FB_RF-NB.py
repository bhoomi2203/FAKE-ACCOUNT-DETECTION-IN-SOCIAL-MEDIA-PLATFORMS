"""
Hybrid Balanced Random Forest + Multinomial Naive Bayes Fake Facebook Account Detection System
Uses Balanced Random Forest for CSV features and Multinomial Naive Bayes for NLP features

Supports two account types with conditional feature selection:
  - Personal accounts (personal(0)/page(1) == 0):
      Uses all numerical features EXCEPT: #followers, #following, category
  - Page accounts (personal(0)/page(1) == 1):
      Uses all numerical features EXCEPT: private, #friends, friends visibility

Data format expected:
  - CSV: rows with NaN in 'fake' column are treated as the unlabelled test set
  - JSON: contains only the labelled entries (matching non-NaN CSV rows)
"""

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Feature definitions ────────────────────────────────────────────────────────

ALL_CSV_FEATURES = [
    'profile pic', 'cover pic', 'fullname characters', 'nums/character fullname',
    'bio length', 'external url', 'private', 'personal(0)/page(1)',
    '#friends', '#followers', '#following', 'friends visibility',
    'category', 'workplace', 'education'
]

PERSONAL_EXCLUDE = {'#followers', '#following', 'category'}
PAGE_EXCLUDE     = {'private', '#friends', 'friends visibility'}

PERSONAL_FEATURES = [f for f in ALL_CSV_FEATURES if f not in PERSONAL_EXCLUDE]
PAGE_FEATURES     = [f for f in ALL_CSV_FEATURES if f not in PAGE_EXCLUDE]


def split_by_type(df):
    """Return boolean masks for personal and page accounts."""
    personal_mask = df['personal(0)/page(1)'] == 0
    page_mask     = df['personal(0)/page(1)'] == 1
    return personal_mask, page_mask


class HybridBRFNBDetector:
    def __init__(self):
        self.brf_personal    = None
        self.brf_page        = None
        self.nb_model        = None
        self.scaler_personal = StandardScaler()
        self.scaler_page     = StandardScaler()
        self.vectorizer      = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self, csv_path='CSV.csv', json_path='NLP.json'):
        """
        Load and separate labelled training data from unlabelled test data.

        The CSV may contain rows where 'fake' is NaN — these are the test
        accounts to predict on. The JSON contains only the labelled entries.
        """
        print("Loading data...")

        csv_data = pd.read_csv(csv_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # ── Split labelled vs unlabelled rows ─────────────────────────────────
        labelled_mask   = csv_data['fake'].notna()
        unlabelled_mask = ~labelled_mask

        labelled_csv   = csv_data[labelled_mask].reset_index(drop=True)
        unlabelled_csv = csv_data[unlabelled_mask].reset_index(drop=True)

        print(f"Labelled samples   : {labelled_mask.sum()}")
        print(f"Unlabelled samples : {unlabelled_mask.sum()}")

        if len(json_data) != len(labelled_csv):
            raise ValueError(
                f"Mismatch: {len(json_data)} JSON entries but "
                f"{len(labelled_csv)} labelled CSV rows."
            )

        y = labelled_csv['fake'].astype(int).values

        # ── NLP features — same for all accounts ─────────────────────────────
        def build_text(entry):
            text  = f"{entry.get('fullname', '')} {entry.get('bio', '')} "
            text += f"{entry.get('work', '')} {entry.get('education', '')} "
            text += ' '.join(entry.get('categories', []))
            text += ' '.join(entry.get('captions', []))
            text += ' '.join(entry.get('comments', []))
            return text.strip() if text.strip() else "empty"

        X_nlp = [build_text(e) for e in json_data]

        # ── CSV feature sets split by account type ────────────────────────────
        personal_mask, page_mask = split_by_type(labelled_csv)

        X_personal       = labelled_csv.loc[personal_mask, PERSONAL_FEATURES].values
        y_personal       = y[personal_mask.values]
        nlp_personal_idx = np.where(personal_mask.values)[0]

        X_page           = labelled_csv.loc[page_mask, PAGE_FEATURES].values
        y_page           = y[page_mask.values]
        nlp_page_idx     = np.where(page_mask.values)[0]

        print(f"Personal accounts  : {personal_mask.sum()}")
        print(f"Page accounts      : {page_mask.sum()}")

        return (X_personal, y_personal, nlp_personal_idx,
                X_page,     y_page,     nlp_page_idx,
                X_nlp, y, unlabelled_csv)

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess_csv(self, X, account_type, fit=True):
        scaler = self.scaler_personal if account_type == 'personal' else self.scaler_page
        return scaler.fit_transform(X) if fit else scaler.transform(X)

    def preprocess_nlp(self, texts, fit=True):
        return self.vectorizer.fit_transform(texts) if fit else self.vectorizer.transform(texts)

    # ── Training ──────────────────────────────────────────────────────────────

    def _train_brf(self, X_train, y_train, label):
        print(f"\nTraining Balanced Random Forest [{label}]...")
        model = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            sampling_strategy='auto',
            replacement=True
        )
        model.fit(X_train, y_train)
        print(f"BRF [{label}] training completed!")
        return model

    def _print_feature_importance(self, model, feature_names, label, top_n=5):
        importance = model.feature_importances_
        top_n = min(top_n, len(feature_names))
        print(f"\nTop {top_n} Most Important Features [{label}]:")
        for idx in np.argsort(importance)[-top_n:][::-1]:
            print(f"  {feature_names[idx]}: {importance[idx]:.4f}")

    def train_nb(self, X_train, y_train):
        print("\nTraining Multinomial Naive Bayes model...")
        self.nb_model = MultinomialNB(alpha=1.0, fit_prior=True)
        self.nb_model.fit(X_train, y_train)
        print("Multinomial Naive Bayes training completed!")

    def train(self, csv_path='CSV.csv', json_path='NLP.json', test_size=0.2):
        """Full training pipeline — single global 80:20 split on all 120 samples."""
        (X_personal, y_personal, nlp_personal_idx,
         X_page,     y_page,     nlp_page_idx,
         X_nlp_list, y_all, unlabelled_csv) = self.load_data(csv_path, json_path)

        # ── Single global 80:20 split: 96 train / 24 test ────────────────────
        global_tr, global_te = train_test_split(
            np.arange(len(y_all)), test_size=test_size,
            random_state=42, stratify=y_all
        )

        # ── Route global indices to per-type local positions ──────────────────
        p_idx_map  = {g: l for l, g in enumerate(nlp_personal_idx)}
        pg_idx_map = {g: l for l, g in enumerate(nlp_page_idx)}

        p_tr  = np.array([p_idx_map[i]  for i in global_tr if i in p_idx_map])
        p_te  = np.array([p_idx_map[i]  for i in global_te if i in p_idx_map])
        pg_tr = np.array([pg_idx_map[i] for i in global_tr if i in pg_idx_map])
        pg_te = np.array([pg_idx_map[i] for i in global_te if i in pg_idx_map])

        # Global indices for NLP alignment per type
        p_te_global  = [i for i in global_te if i in p_idx_map]
        pg_te_global = [i for i in global_te if i in pg_idx_map]

        # ── Train Personal BRF ────────────────────────────────────────────────
        X_p_train = self.preprocess_csv(X_personal[p_tr], 'personal', fit=True)
        X_p_test  = self.preprocess_csv(X_personal[p_te], 'personal', fit=False)
        self.brf_personal = self._train_brf(X_p_train, y_personal[p_tr], 'Personal')
        self._print_feature_importance(self.brf_personal, PERSONAL_FEATURES, 'Personal')

        # ── Train Page BRF ────────────────────────────────────────────────────
        X_pg_train = self.preprocess_csv(X_page[pg_tr], 'page', fit=True)
        X_pg_test  = self.preprocess_csv(X_page[pg_te], 'page', fit=False)
        self.brf_page = self._train_brf(X_pg_train, y_page[pg_tr], 'Page')
        self._print_feature_importance(self.brf_page, PAGE_FEATURES, 'Page')

        # ── Train NB on same global split ─────────────────────────────────────
        X_nlp_train = self.preprocess_nlp([X_nlp_list[i] for i in global_tr], fit=True)
        self.train_nb(X_nlp_train, y_all[global_tr])

        # ── Hybrid predictions on the 24 test samples ─────────────────────────
        nlp_p_test  = self.preprocess_nlp([X_nlp_list[i] for i in p_te_global],  fit=False)
        nlp_pg_test = self.preprocess_nlp([X_nlp_list[i] for i in pg_te_global], fit=False)

        preds_p,  conf_p  = self.predict(X_p_test,  nlp_p_test,  'personal')
        preds_pg, conf_pg = self.predict(X_pg_test, nlp_pg_test, 'page')

        # Combine all 24 test results in global_te order
        all_preds = np.concatenate([preds_p,             preds_pg])
        all_conf  = np.concatenate([conf_p,              conf_pg])
        all_true  = np.concatenate([y_personal[p_te],    y_page[pg_te]])

        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        self._print_metrics(all_true, all_preds)
        self._show_sample_predictions(all_preds, all_conf, all_true)

        # Save
        self.save_models()

        return unlabelled_csv

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X_csv_scaled, X_nlp_tfidf, account_type, return_confidence=True):
        """Hybrid prediction: BRF 40% + NB 60%."""
        brf = self.brf_personal if account_type == 'personal' else self.brf_page
        brf_proba = brf.predict_proba(X_csv_scaled)[:, 1]
        nb_proba  = self.nb_model.predict_proba(X_nlp_tfidf)[:, 1]

        combined    = 0.4 * brf_proba + 0.6 * nb_proba
        predictions = (combined >= 0.5).astype(int)

        if return_confidence:
            confidence = np.where(predictions == 1, combined * 100, (1 - combined) * 100)
            return predictions, confidence
        return predictions

    def predict_single(self, csv_features_dict, text_data):
        """
        Predict a single account.

        csv_features_dict : dict keyed by column name, must include 'personal(0)/page(1)'
        text_data         : raw text string (bio, captions, comments, etc.)
        """
        account_type_val = csv_features_dict.get('personal(0)/page(1)', 0)
        account_type     = 'personal' if account_type_val == 0 else 'page'
        feature_cols     = PERSONAL_FEATURES if account_type == 'personal' else PAGE_FEATURES

        csv_array  = np.array([csv_features_dict[col] for col in feature_cols]).reshape(1, -1)
        scaler     = self.scaler_personal if account_type == 'personal' else self.scaler_page
        csv_scaled = scaler.transform(csv_array)

        text_data     = text_data.strip() if text_data.strip() else "empty"
        nlp_processed = self.vectorizer.transform([text_data])

        prediction, confidence = self.predict(csv_scaled, nlp_processed, account_type)

        result = "Fake" if prediction[0] == 1 else "Real"
        print(f"\nAccount type : {'Personal' if account_type == 'personal' else 'Page'}")
        print(f"Prediction   : {result} ({confidence[0]:.1f}% confidence)")
        return result, confidence[0]

    # ── Evaluation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _print_metrics(y_true, y_pred):
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        accuracy  = accuracy_score(y_true, y_pred)
        cm        = confusion_matrix(y_true, y_pred)
        print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Real  Fake")
        print(f"Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")

    @staticmethod
    def _show_sample_predictions(predictions, confidence, y_test, n=10):
        print(f"\nSample Predictions:")
        print("-" * 40)
        for i in range(min(n, len(predictions))):
            result = "Fake" if predictions[i] == 1 else "Real"
            actual = "Fake" if y_test[i]     == 1 else "Real"
            status = "✓" if predictions[i] == y_test[i] else "✗"
            print(f"{status} {result} ({confidence[i]:.1f}% confidence) - Actual: {actual}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_models(self,
                    brf_personal_path='brf_personal_model.pkl',
                    brf_page_path='brf_page_model.pkl',
                    nb_path='nb_model.pkl',
                    scaler_personal_path='scaler_personal.pkl',
                    scaler_page_path='scaler_page.pkl',
                    vectorizer_path='vectorizer.pkl'):
        print("\nSaving models...")
        for obj, path in [
            (self.brf_personal,    brf_personal_path),
            (self.brf_page,        brf_page_path),
            (self.nb_model,        nb_path),
            (self.scaler_personal, scaler_personal_path),
            (self.scaler_page,     scaler_page_path),
            (self.vectorizer,      vectorizer_path),
        ]:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        print("Models saved successfully!")

    def load_models(self,
                    brf_personal_path='brf_personal_model.pkl',
                    brf_page_path='brf_page_model.pkl',
                    nb_path='nb_model.pkl',
                    scaler_personal_path='scaler_personal.pkl',
                    scaler_page_path='scaler_page.pkl',
                    vectorizer_path='vectorizer.pkl'):
        print("Loading models...")
        with open(brf_personal_path,   'rb') as f: self.brf_personal    = pickle.load(f)
        with open(brf_page_path,       'rb') as f: self.brf_page         = pickle.load(f)
        with open(nb_path,             'rb') as f: self.nb_model         = pickle.load(f)
        with open(scaler_personal_path,'rb') as f: self.scaler_personal  = pickle.load(f)
        with open(scaler_page_path,    'rb') as f: self.scaler_page      = pickle.load(f)
        with open(vectorizer_path,     'rb') as f: self.vectorizer       = pickle.load(f)
        print("Models loaded successfully!")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("HYBRID BALANCED RANDOM FOREST + MULTINOMIAL NAIVE BAYES")
    print("FAKE FACEBOOK ACCOUNT DETECTION SYSTEM")
    print("="*60)

    detector = HybridBRFNBDetector()

    try:
        unlabelled_csv = detector.train('CSV.csv', 'NLP.json', test_size=0.2)
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)

        # ── Example: Personal account prediction ──────────────────────────────
        print("\n" + "="*60)
        print("EXAMPLE: Personal Account Prediction")
        print("="*60)
        personal_example = {
            'profile pic':             1,
            'cover pic':               0,
            'fullname characters':     10,
            'nums/character fullname': 0.0,
            'bio length':              0,
            'external url':            0,
            'private':                 1,
            'personal(0)/page(1)':     0,   # ← personal
            '#friends':              120,
            'friends visibility':      1,
            'workplace':               0,
            'education':               1,
        }
        personal_text = "Hey! Just a normal person sharing life moments."
        detector.predict_single(personal_example, personal_text)

        # ── Example: Page account prediction ──────────────────────────────────
        print("\n" + "="*60)
        print("EXAMPLE: Page Account Prediction")
        print("="*60)
        page_example = {
            'profile pic':             1,
            'cover pic':               1,
            'fullname characters':     18,
            'nums/character fullname': 0.1,
            'bio length':              80,
            'external url':            1,
            'personal(0)/page(1)':     1,   # ← page
            '#followers':           5000,
            '#following':            300,
            'category':                1,
            'workplace':               0,
            'education':               0,
        }
        page_text = "Buy followers now! Click here for instant fame! 🔥🔥🔥"
        detector.predict_single(page_example, page_text)

    except Exception as e:
        import traceback
        print(f"\nError during training: {str(e)}")
        traceback.print_exc()
        print("\nPlease ensure CSV.csv and NLP.json are in the same directory.")