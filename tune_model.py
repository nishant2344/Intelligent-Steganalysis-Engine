import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

def tune_and_compare(csv_path, model_output_dir):
    print("Loading extracted DLP audio features...")
    df = pd.read_csv(csv_path).dropna()
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define stronger candidate models
    models = {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        )
    }

    best_model = None
    best_name = ""
    best_acc = 0.0
    best_preds = None

    print("\n--- Benchmarking Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        print(f"  -> {name} Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            best_preds = preds

    target_names = ['CLEAN (0)', 'STANDARD_LSB (1)', 'PHASE_ENCODED (2)']
    print("\n" + "="*55)
    print(f"WINNER: {best_name} ({best_acc:.2f}% Accuracy)")
    print("="*55)
    print("DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, best_preds, target_names=target_names))
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, best_preds))
    print("="*55)

    # Save the winning model
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    model_path = os.path.join(model_output_dir, 'stego_detector_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nSaved winning model ({best_name}) to: {model_path}")

if __name__ == "__main__":
    feature_csv = r"D:\FINAL_PROJECT\ISE\data\audio_features.csv"
    model_directory = r"D:\FINAL_PROJECT\ISE\models"
    tune_and_compare(feature_csv, model_directory)