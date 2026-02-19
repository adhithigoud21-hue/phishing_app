import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Shared feature extractor ─────────────────────────────────────────────────
from feature_extractor import extract_features, FEATURE_NAMES


class AdvancedPhishingDetector:
    def __init__(self, data_path='PhiUSIIL_Phishing_URL_Dataset.csv', min_samples=10000):
        self.data_path       = data_path
        self.min_samples     = min_samples
        self.model           = None
        self.scaler          = StandardScaler()
        self.feature_names   = FEATURE_NAMES
        self.feature_importance = None
        self.metrics         = {}

    # ── 1. Load & prepare ────────────────────────────────────────────────────
    def load_and_prepare_data(self):
        print("=" * 70)
        print("📊 LOADING PHISHING DETECTION DATASET")
        print("=" * 70)

        df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")

        # Find URL column
        url_col = None
        for c in ['url', 'URL', 'Url', 'link', 'Link']:
            if c in df.columns:
                url_col = c
                break
        if url_col is None:
            for c in df.columns:
                if df[c].dtype == object:
                    url_col = c
                    break
        if url_col is None:
            raise ValueError("Cannot find a URL column in the dataset!")
        print(f"✅ URL column  : '{url_col}'")

        # Find label column
        label_col = None
        for c in ['label', 'Label', 'status', 'Status', 'class',
                  'Class', 'target', 'Target', 'phishing', 'type']:
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            label_col = df.columns[-1]
            print(f"⚠️  Assuming '{label_col}' is the label column")
        else:
            print(f"✅ Label column: '{label_col}'")

        # Sample rows
        sample_size = min(len(df), self.min_samples * 2)
        df = df[[url_col, label_col]].dropna().sample(n=sample_size, random_state=42)
        print(f"📦 Using {len(df):,} rows for feature extraction")

        # Convert labels → 0 / 1
        y_raw = df[label_col]
        if y_raw.dtype == object:
            label_map = {}
            for lbl in y_raw.unique():
                lo = str(lbl).lower()
                label_map[lbl] = 1 if any(t in lo for t in
                    ['phish', 'bad', 'malicious', 'fraud', '1']) else 0
            y = y_raw.map(label_map)
            print(f"✅ Label mapping: {label_map}")
        else:
            y = y_raw.astype(int)

        print(f"\n📊 Class Distribution:")
        print(f"   Legitimate : {(y==0).sum():,}  ({(y==0).sum()/len(y)*100:.1f}%)")
        print(f"   Phishing   : {(y==1).sum():,}  ({(y==1).sum()/len(y)*100:.1f}%)")

        # Extract features from URLs
        print(f"\n🔍 Extracting features from {len(df):,} URLs...")
        urls  = df[url_col].tolist()
        total = len(urls)
        rows  = []
        for i, url in enumerate(urls):
            if i % 2000 == 0:
                print(f"   Progress: {i:,}/{total:,}  ({i/total*100:.0f}%)")
            feat = extract_features(url)
            rows.append([feat.get(fn, 0) for fn in FEATURE_NAMES])

        X = pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0)
        print(f"✅ Done — {len(FEATURE_NAMES)} features per URL")

        return X, y.reset_index(drop=True)

    # ── 2. Train ─────────────────────────────────────────────────────────────
    def train_ensemble_model(self, X, y):
        print("\n" + "=" * 70)
        print("🎯 TRAINING ADVANCED ENSEMBLE MODEL")
        print("=" * 70)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"✅ Training set : {len(X_train):,} | Test set : {len(X_test):,}")

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        print("\n🔧 Initializing models...")
        rf  = RandomForestClassifier(n_estimators=200, max_depth=20,
                                     min_samples_split=5, random_state=42,
                                     n_jobs=-1, class_weight='balanced')
        gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                         max_depth=7, random_state=42, subsample=0.8)
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=-1, eval_metric='logloss', verbosity=0)

        print("🤖 Creating Voting Classifier ensemble...")
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)],
            voting='soft', n_jobs=-1)

        print("🎓 Training... (a few minutes)")
        ensemble.fit(X_train_s, y_train)
        print("✅ Training complete!")
        self.model = ensemble

        # Evaluate
        print("\n" + "=" * 70)
        print("📈 MODEL EVALUATION")
        print("=" * 70)

        y_pred       = ensemble.predict(X_test_s)
        y_pred_proba = ensemble.predict_proba(X_test_s)[:, 1]
        train_acc    = accuracy_score(y_train, ensemble.predict(X_train_s))

        self.metrics = {
            'accuracy':       accuracy_score(y_test, y_pred),
            'precision':      precision_score(y_test, y_pred),
            'recall':         recall_score(y_test, y_pred),
            'f1_score':       f1_score(y_test, y_pred),
            'roc_auc':        roc_auc_score(y_test, y_pred_proba),
            'train_accuracy': train_acc,
        }

        print(f"🎯 Training Accuracy : {train_acc*100:.2f}%")
        print(f"\n📊 Test Set Performance:")
        for k, v in self.metrics.items():
            if k != 'train_accuracy':
                print(f"   ✅ {k:12s}: {v*100:.2f}%")

        print("\n" + "=" * 70)
        print("📋 CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_test, y_pred,
                                    target_names=['Legitimate', 'Phishing']))

        cm = confusion_matrix(y_test, y_pred)
        print("📊 Confusion Matrix:")
        print(f"   True Negatives  : {cm[0,0]:,}")
        print(f"   False Positives : {cm[0,1]:,}")
        print(f"   False Negatives : {cm[1,0]:,}")
        print(f"   True Positives  : {cm[1,1]:,}")

        rf_comp = ensemble.named_estimators_['rf']
        self.feature_importance = pd.DataFrame({
            'feature':    FEATURE_NAMES,
            'importance': rf_comp.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n📊 Top 15 Important Features:")
        for _, row in self.feature_importance.head(15).iterrows():
            print(f"   {row['feature']:35s}: {row['importance']:.4f}")

        return X_test_s, y_test, y_pred, y_pred_proba

    # ── 3. Save ──────────────────────────────────────────────────────────────
    def save_model(self):
        print("\n" + "=" * 70)
        print("💾 SAVING MODEL FILES")
        print("=" * 70)

        with open('phishing_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("✅ phishing_model.pkl")

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✅ scaler.pkl")

        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(FEATURE_NAMES, f)
        print("✅ feature_names.pkl")

        with open('model_metrics.pkl', 'wb') as f:
            pickle.dump(self.metrics, f)
        print("✅ model_metrics.pkl")

        self.feature_importance.to_csv('feature_importance.csv', index=False)
        print("✅ feature_importance.csv")

    # ── 4. Visualise ─────────────────────────────────────────────────────────
    def create_visualizations(self, y_test, y_pred, y_pred_proba):
        print("\n📊 Creating visualizations...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'AUC = {self.metrics["roc_auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(alpha=0.3)

        top = self.feature_importance.head(15)
        axes[1, 0].barh(range(len(top)), top['importance'], color='steelblue')
        axes[1, 0].set_yticks(range(len(top)))
        axes[1, 0].set_yticklabels(top['feature'], fontsize=9)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='x', alpha=0.3)

        names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [self.metrics['accuracy'], self.metrics['precision'],
                  self.metrics['recall'],   self.metrics['f1_score'],
                  self.metrics['roc_auc']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars   = axes[1, 1].bar(names, values, color=colors)
        axes[1, 1].set_ylim([0, 1.15])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2,
                            val + 0.02, f'{val:.3f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ model_performance.png saved")

    # ── 5. Full pipeline ─────────────────────────────────────────────────────
    def train(self):
        start = datetime.now()
        X, y = self.load_and_prepare_data()
        X_test_s, y_test, y_pred, y_pred_proba = self.train_ensemble_model(X, y)
        self.create_visualizations(y_test, y_pred, y_pred_proba)
        self.save_model()

        elapsed = (datetime.now() - start).total_seconds()
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Total time : {elapsed:.0f} seconds")
        print(f"🎯 Accuracy   : {self.metrics['accuracy']*100:.2f}%")
        print(f"🎯 F1-Score   : {self.metrics['f1_score']*100:.2f}%")
        print(f"🎯 ROC-AUC    : {self.metrics['roc_auc']*100:.2f}%")
        print("✅ Run: streamlit run app_streamlit.py")
        print("=" * 70)


if __name__ == '__main__':
    detector = AdvancedPhishingDetector(
        data_path='PhiUSIIL_Phishing_URL_Dataset.csv',
        min_samples=10000
    )
    detector.train()