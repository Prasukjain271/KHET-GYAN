import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# =========================================================
# 1️⃣ LOAD DATA
# =========================================================

df = pd.read_csv("/Users/apple/Desktop/hackathon_ml/data/Crop_production_final.csv")

print("Dataset Shape:", df.shape)
print("Unique Crops:", df['Crop'].nunique())

# =========================================================
# 2️⃣ DEFINE TARGET
# =========================================================

y = df['Crop']

# =========================================================
# 3️⃣ DEFINE FEATURE GROUPS
# =========================================================

base_features = ['N', 'P', 'K', 'rainfall', 'temperature']
ph_feature = ['pH']
state_feature = ['State_Name']
season_feature = ['Crop_Type']

# =========================================================
# 4️⃣ TRAINING FUNCTION
# =========================================================

def train_model(feature_columns, model_name):

    print(f"\n==============================")
    print(f"Training {model_name}")
    print(f"Using features: {feature_columns}")
    print(f"==============================")

    X = df[feature_columns].copy()

    # Identify categorical & numerical columns
    categorical_cols = [col for col in feature_columns 
                        if col in ['State_Name', 'Crop_Type']]
    
    numerical_cols = [col for col in feature_columns 
                      if col not in categorical_cols]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Full ML pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"Weighted F1 Score: {weighted_f1:.4f}")

    return model, weighted_f1

# =========================================================
# 5️⃣ TRAIN MULTIPLE MODEL VARIANTS
# =========================================================

results = {}

# Model 1: Soil + Weather
features_1 = base_features
model1, f1_1 = train_model(features_1, "Model 1: Soil + Weather")
results["Model1"] = (model1, f1_1)

# Model 2: + State
features_2 = base_features + state_feature
model2, f1_2 = train_model(features_2, "Model 2: + State")
results["Model2"] = (model2, f1_2)

# Model 3: + State + pH
features_3 = base_features + state_feature + ph_feature
model3, f1_3 = train_model(features_3, "Model 3: + State + pH")
results["Model3"] = (model3, f1_3)

# Model 4: + State + pH + Season
features_4 = base_features + state_feature + ph_feature + season_feature
model4, f1_4 = train_model(features_4, "Model 4: + State + pH + Season")
results["Model4"] = (model4, f1_4)

# =========================================================
# 6️⃣ SELECT BEST MODEL
# =========================================================

best_model_name = max(results, key=lambda x: results[x][1])
best_model, best_f1 = results[best_model_name]

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Best Weighted F1: {best_f1:.4f}")
print("==============================")

# =========================================================
# 7️⃣ SAVE BEST MODEL
# =========================================================

joblib.dump(best_model, "best_crop_recommendation_model.pkl")

print("\nBest model saved as best_crop_recommendation_model.pkl")


from sklearn.metrics import top_k_accuracy_score
import numpy as np

# Use best model
model = best_model

# Recreate train-test split using best features
X = df[features_4].copy()
y = df['Crop']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Fit again to ensure consistency
model.fit(X_train, y_train)

# Get probabilities
y_proba = model.predict_proba(X_test)

# Top-5 accuracy
top5_acc = top_k_accuracy_score(
    y_test,
    y_proba,
    k=5,
    labels=model.classes_
)

print(f"\nTop-5 Accuracy: {top5_acc:.4f}")


from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring='f1_weighted',
    n_jobs=-1
)

print("\nCross-Validation Weighted F1 Scores:")
print(cv_scores)
print(f"\nMean CV F1: {cv_scores.mean():.4f}")
print(f"Std Deviation: {cv_scores.std():.4f}")
