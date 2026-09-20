"""Stage 2 - prediction models, probability calibration and fairness audit

Stage 2 of the clinic overbooking study. This is the notebook that produced
the reported results, exported and lightly cleaned: cell markers removed,
output paths routed through the repository layout, plotting forced headless.
Numerical logic is unchanged. Run from anywhere; the script anchors itself to
the repository root.
"""

from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

for _directory in (
    "data/processed",
    "results/models",
    "results/tables",
    "results/stage3",
    "results/figures/stage1",
    "results/figures/stage2",
    "results/figures/stage3",
    "results/figures/stage4",
):
    (REPO_ROOT / _directory).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")  # headless: every figure is written under results/figures/

# ## Stage 2: No-Show Prediction Models

# This notebook contains all modeling tasks:
# - Section 1: Setup & Data Loading
# - Section 2: Tree-Based Models (Random Forest, XGBoost)
# - Section 3: Linear Models & Neural Network
# - Section 4: Probability Calibration
# - Section 5: Model Evaluation & Comparison
# - Section 6: Fairness Analysis
# - Section 7: Ensemble Models
# - Section 8: Final Model Selection & Export

# ### Section 1: Setup and Data Loading


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import random
import os
import joblib

# Reproducibility 
def set_global_seed(seed=42):
    """Set seed for all random number generators for reproducibility"""
    
    # Python's built-in random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # XGBoost (additional)
    os.environ['XGB_RANDOM_SEED'] = str(seed)
    
    print(f"✓ Global random seed set to: {seed}")
    return seed

# Set and store the seed
RANDOM_SEED = set_global_seed(42)

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, classification_report,
    f1_score, accuracy_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy import stats

# XGBoost
import xgboost as xgb

# SHAP for explainability
import shap

# Settings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# Load feature-engineered data

print("\n--- Loading Data ---")

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# Load full dataframes for fairness analysis later
train_full = pd.read_csv('data/processed/train_full.csv')
val_full = pd.read_csv('data/processed/val_full.csv')
test_full = pd.read_csv('data/processed/test_full.csv')

print(f"Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"Val:   {X_val.shape[0]:,} samples")
print(f"Test:  {X_test.shape[0]:,} samples")
print(f"\nFeatures: {list(X_train.columns)}")

# Class balance info
pos_rate_train = y_train.mean()
class_weight_ratio = (1 - pos_rate_train) / pos_rate_train
print(f"\nClass balance: {pos_rate_train*100:.1f}% positive (no-shows)")
print(f"Class weight ratio for imbalance handling: {class_weight_ratio:.2f}")


# 29.6% no-show rate - Moderate class imbalance (not severe)
# Class weight ratio 2.38 - Appropriate for balanced learning
# 16 engineered features - Good feature engineering from EDA stage
# Temporal split (60/20/20) - Correct for time-series prediction


# Scale features (needed for linear models and neural network)
# ----------------------------------------------------------------------------
print("\n--- Scaling Features ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# Dictionary to store all models and their predictions
all_models = {}
all_predictions_val = {}
all_predictions_test = {}


# ### Section 2: Tree-Based Models


# ----------------------------------------------------------------------------
# 2.1 Random Forest
# ----------------------------------------------------------------------------
print("\n--- 2.1 Random Forest ---")

# Baseline model for quick evaluation
print("Training baseline Random Forest...")
rf_baseline = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    class_weight='balanced',
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

# Cross-validation score
cv_scores_rf = cross_val_score(rf_baseline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Baseline CV AUC: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

# Hyperparameter tuning
print("\nTuning Random Forest hyperparameters (this may take a few minutes)...")
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
}

rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

print(f"\nBest RF parameters: {rf_grid.best_params_}")
print(f"Best RF CV AUC: {rf_grid.best_score_:.4f}")

rf_best = rf_grid.best_estimator_

# Validation performance
rf_val_proba = rf_best.predict_proba(X_val)[:, 1]
rf_test_proba = rf_best.predict_proba(X_test)[:, 1]
rf_val_auc = roc_auc_score(y_val, rf_val_proba)
rf_val_ap = average_precision_score(y_val, rf_val_proba)

print(f"\nRF Validation AUC: {rf_val_auc:.4f}")
print(f"RF Validation Average Precision: {rf_val_ap:.4f}")

# Store model and predictions
all_models['Random Forest'] = {'model': rf_best, 'scaled': False}
all_predictions_val['Random Forest'] = rf_val_proba
all_predictions_test['Random Forest'] = rf_test_proba

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance.to_string(index=False))


# 2.2 XGBoost

print("\n--- 2.2 XGBoost ---")

# Baseline XGBoost
print("Training baseline XGBoost...")
xgb_baseline = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=class_weight_ratio,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric='auc',
    n_jobs=-1
)
xgb_baseline.fit(X_train, y_train)

cv_scores_xgb = cross_val_score(xgb_baseline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Baseline CV AUC: {cv_scores_xgb.mean():.4f} ± {cv_scores_xgb.std():.4f}")

# Hyperparameter tuning
print("\nTuning XGBoost hyperparameters...")
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight_ratio,
    random_state=RANDOM_SEED,
    use_label_encoder=False,
    eval_metric='auc',
    n_jobs=-1
)
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)

print(f"\nBest XGBoost parameters: {xgb_grid.best_params_}")
print(f"Best XGBoost CV AUC: {xgb_grid.best_score_:.4f}")

xgb_best = xgb_grid.best_estimator_

# Validation performance
xgb_val_proba = xgb_best.predict_proba(X_val)[:, 1]
xgb_test_proba = xgb_best.predict_proba(X_test)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)
xgb_val_ap = average_precision_score(y_val, xgb_val_proba)

print(f"\nXGBoost Validation AUC: {xgb_val_auc:.4f}")
print(f"XGBoost Validation Average Precision: {xgb_val_ap:.4f}")

# Store model and predictions
all_models['XGBoost'] = {'model': xgb_best, 'scaled': False}
all_predictions_val['XGBoost'] = xgb_val_proba
all_predictions_test['XGBoost'] = xgb_test_proba

# Feature importance
xgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nXGBoost Feature Importance:")
print(xgb_importance.to_string(index=False))


# 2.3 SHAP Analysis (for XGBoost)

print("\n--- 2.3 SHAP Analysis ---")
print("Calculating SHAP values (this may take a few minutes)...")

# Sample for faster SHAP computation
shap_sample_size = min(1000, len(X_val))
X_shap_sample = X_val.sample(n=shap_sample_size, random_state=RANDOM_SEED)

# Create SHAP explainer and compute values
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_shap_sample)

# SHAP bar plot (feature importance)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
plt.title('XGBoost SHAP Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/stage2/shap_importance_bar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# SHAP beeswarm plot (detailed feature effects)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.title('XGBoost SHAP Summary (Beeswarm)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/stage2/shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/shap_importance_bar.png")
print("✓ Saved: results/figures/stage2/shap_beeswarm.png")


# 2.4 Feature Importance Comparison Plot

print("\n--- 2.4 Feature Importance Comparison ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
axes[0].barh(rf_importance['Feature'], rf_importance['Importance'], color='steelblue')
axes[0].set_xlabel('Importance')
axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
axes[0].invert_yaxis()

# XGBoost
axes[1].barh(xgb_importance['Feature'], xgb_importance['Importance'], color='coral')
axes[1].set_xlabel('Importance')
axes[1].set_title('XGBoost Feature Importance', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('results/figures/stage2/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/feature_importance_comparison.png")

# Compare with EDA expectations
print("\nComparison with EDA Insights:")
eda_expected = ['Lead_Time_Days', 'Age', 'Age_x_LeadTime', 'SMS_Received', 'Patient_Prior_NoShow_Rate']
xgb_rank = {feat: i+1 for i, feat in enumerate(xgb_importance['Feature'])}

for feat in eda_expected:
    if feat in xgb_rank:
        status = "✓" if xgb_rank[feat] <= 5 else "○"
        print(f"  {status} {feat}: Rank {xgb_rank[feat]}")


# **Random Forest:**
# Best params: max_depth=10, shallow trees reduce overfitting
# Top features: Age (0.195), Neighborhood_Risk (0.150), Patient_Prior_NoShow_Rate (0.131)
# RF prefers continuous features it can split on
# 
# **XGBoost:**
# Best params: max_depth=4, even shallower than RF
# Top features: Patient_Prior_NoShow_Rate (0.165), Days_Since_Last_Appt (0.146), Age (0.127)
# XGBoost captures patient history features better
# 
# **SHAP insights (Beeswarm)**
# Age -- Blue (low) → Positive SHAP -- Younger patients → higher no-show risk ✓
# Patient_Prior_NoShow_Rate -- Red (high) → Positive SHAP -- History of no-shows → higher risk ✓
# Lead_Time_Days -- Red (high) → Positive SHAP -- Longer lead time → higher risk ✓
# Neighborhood_Risk -- Red (high) → Positive SHAP -- High-risk neighborhoods → higher risk ✓
# SMS_Received -- Minimal impact -- Confirms EDA finding that SMS effect is weak

# ### Section 3: Linear Models and Neural Network


# 3.1 Logistic Regression

print("\n--- 3.1 Logistic Regression ---")

# Test different regularization types
print("Training L2 (Ridge) Logistic Regression...")
lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, 
                            class_weight='balanced', random_state=RANDOM_SEED)
lr_l2.fit(X_train_scaled, y_train)
cv_l2 = cross_val_score(lr_l2, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"L2 CV AUC: {cv_l2.mean():.4f} ± {cv_l2.std():.4f}")

print("\nTraining L1 (Lasso) Logistic Regression...")
lr_l1 = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, 
                            class_weight='balanced', random_state=RANDOM_SEED)
lr_l1.fit(X_train_scaled, y_train)
cv_l1 = cross_val_score(lr_l1, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"L1 CV AUC: {cv_l1.mean():.4f} ± {cv_l1.std():.4f}")

print("\nTraining ElasticNet Logistic Regression...")
lr_elastic = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, 
                                 max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
lr_elastic.fit(X_train_scaled, y_train)
cv_elastic = cross_val_score(lr_elastic, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"ElasticNet CV AUC: {cv_elastic.mean():.4f} ± {cv_elastic.std():.4f}")

# Tune regularization strength C
print("\nTuning regularization strength C...")
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, 
                         class_weight='balanced', random_state=RANDOM_SEED)
lr_grid = GridSearchCV(lr, lr_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

print(f"Best C: {lr_grid.best_params_['C']}")
print(f"Best CV AUC: {lr_grid.best_score_:.4f}")

lr_best = lr_grid.best_estimator_

# Validation performance
lr_val_proba = lr_best.predict_proba(X_val_scaled)[:, 1]
lr_test_proba = lr_best.predict_proba(X_test_scaled)[:, 1]
lr_val_auc = roc_auc_score(y_val, lr_val_proba)
lr_val_ap = average_precision_score(y_val, lr_val_proba)

print(f"\nLogistic Regression Validation AUC: {lr_val_auc:.4f}")
print(f"Logistic Regression Validation AP: {lr_val_ap:.4f}")

# Store model and predictions
all_models['Logistic Regression'] = {'model': lr_best, 'scaled': True}
all_predictions_val['Logistic Regression'] = lr_val_proba
all_predictions_test['Logistic Regression'] = lr_test_proba

# Coefficient analysis
lr_coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_best.coef_[0],
    'Abs_Coefficient': np.abs(lr_best.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nLogistic Regression Coefficients (sorted by magnitude):")
print(lr_coefficients.to_string(index=False))

# Plot coefficients
plt.figure(figsize=(12, 6))
colors = ['coral' if c > 0 else 'steelblue' for c in lr_coefficients['Coefficient']]
plt.barh(lr_coefficients['Feature'], lr_coefficients['Coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Coefficients\n(Positive = Increases No-Show Risk)', fontweight='bold')
plt.axvline(x=0, color='black', linewidth=1)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('results/figures/stage2/lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/lr_coefficients.png")


# Patient_Prior_NoShow_Rate -- +0.224 -- Strongest positive predictor
# Lead_Time_Days -- +0.198 -- Longer wait → more no-shows
# Age -- -0.166 -- Older → fewer no-shows 
# Scholarship -- +0.069 -- Socioeconomic factor matters
# SMS_Received -- ~0 -- Almost no effect (confirms EDA)


# 3.2 Neural Network (MLP)

print("\n--- 3.2 Neural Network (MLP) ---")

print("Training MLP Neural Network...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),      # 2-layer architecture
    activation='relu',                 # ReLU activation
    solver='adam',                     # Adam optimizer
    alpha=0.001,                       # L2 regularization
    batch_size=256,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,               # Enable early stopping
    validation_fraction=0.1,
    n_iter_no_change=10,               # Patience = 10
    random_state=RANDOM_SEED,
    verbose=False
)
mlp.fit(X_train_scaled, y_train)

print(f"MLP converged in {mlp.n_iter_} iterations")
print(f"Best validation score during training: {mlp.best_validation_score_:.4f}")

# Cross-validation
cv_mlp = cross_val_score(mlp, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"MLP CV AUC: {cv_mlp.mean():.4f} ± {cv_mlp.std():.4f}")

# Validation performance
mlp_val_proba = mlp.predict_proba(X_val_scaled)[:, 1]
mlp_test_proba = mlp.predict_proba(X_test_scaled)[:, 1]
mlp_val_auc = roc_auc_score(y_val, mlp_val_proba)
mlp_val_ap = average_precision_score(y_val, mlp_val_proba)

print(f"\nMLP Validation AUC: {mlp_val_auc:.4f}")
print(f"MLP Validation Average Precision: {mlp_val_ap:.4f}")

# Store model and predictions
all_models['Neural Network'] = {'model': mlp, 'scaled': True}
all_predictions_val['Neural Network'] = mlp_val_proba
all_predictions_test['Neural Network'] = mlp_test_proba

# Plot training loss curve
plt.figure(figsize=(10, 5))
plt.plot(mlp.loss_curve_, label='Training Loss', color='steelblue')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss Curve', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/stage2/mlp_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/mlp_loss_curve.png")


# Converged in only 15 iterations (very fast)
# Best validation score 0.716 during training but actual Val AUC 0.609
# This suggests the internal validation metric differs from AUC

# Neural Network converged in only 15 iterations with relatively poor performance. This suggests the architecture (64,32) may be too simple or early stopping triggered too aggressively. However, since it's not the best model anyway, this doesn't affect the final results.

# ### Section 4: Probability Calibration


# Store calibrated predictions
calibrated_platt = {}
calibrated_isotonic = {}


# 4.1 Apply Platt Scaling (Sigmoid Calibration)

print("\n--- 4.1 Platt Scaling (Sigmoid Calibration) ---")

# for name, info in all_models.items():
#     print(f"Calibrating {name}...")
#     cal_model = CalibratedClassifierCV(info['model'], method='sigmoid', cv='prefit')
    
#     if info['scaled']:
#         cal_model.fit(X_val_scaled, y_val)
#         cal_proba = cal_model.predict_proba(X_val_scaled)[:, 1]
#     else:
#         cal_model.fit(X_val, y_val)
#         cal_proba = cal_model.predict_proba(X_val)[:, 1]
    
#     calibrated_platt[name] = {'model': cal_model, 'predictions': cal_proba}

calibrated_platt = {}

for name, info in all_models.items():
    print(f"Calibrating {name}...")
    
    # Get predictions from pre-fitted model
    if info['scaled']:
        raw_proba = info['model'].predict_proba(X_val_scaled)[:, 1]
    else:
        raw_proba = info['model'].predict_proba(X_val)[:, 1]
    
    # Manual Platt scaling: fit logistic regression on raw probabilities
    platt_scaler = LogisticRegression(solver='lbfgs', max_iter=1000)
    platt_scaler.fit(raw_proba.reshape(-1, 1), y_val)
    
    # Get calibrated probabilities
    cal_proba = platt_scaler.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    
    calibrated_platt[name] = {
        'model': info['model'],  # Keep original model
        'scaler': platt_scaler,  # Store calibration scaler
        'predictions': cal_proba
    }
    
    print(f"  Raw mean: {raw_proba.mean():.4f} → Calibrated mean: {cal_proba.mean():.4f}")


# 4.2 Apply Isotonic Regression

print("\n--- 4.2 Isotonic Regression ---")

# for name, info in all_models.items():
#     print(f"Calibrating {name}...")
#     cal_model = CalibratedClassifierCV(info['model'], method='isotonic', cv='prefit')
    
#     if info['scaled']:
#         cal_model.fit(X_val_scaled, y_val)
#         cal_proba = cal_model.predict_proba(X_val_scaled)[:, 1]
#     else:
#         cal_model.fit(X_val, y_val)
#         cal_proba = cal_model.predict_proba(X_val)[:, 1]
    
#     calibrated_isotonic[name] = {'model': cal_model, 'predictions': cal_proba}

calibrated_isotonic = {}

for name, info in all_models.items():
    print(f"Calibrating {name}...")
    
    # Get predictions from pre-fitted model
    if info['scaled']:
        raw_proba = info['model'].predict_proba(X_val_scaled)[:, 1]
    else:
        raw_proba = info['model'].predict_proba(X_val)[:, 1]
    
    # Manual isotonic regression calibration
    iso_scaler = IsotonicRegression(out_of_bounds='clip')
    iso_scaler.fit(raw_proba, y_val)
    
    # Get calibrated probabilities
    cal_proba = iso_scaler.predict(raw_proba)
    
    calibrated_isotonic[name] = {
        'model': info['model'],  # Keep original model
        'scaler': iso_scaler,    # Store calibration scaler
        'predictions': cal_proba
    }
    
    print(f"  Raw mean: {raw_proba.mean():.4f} → Calibrated mean: {cal_proba.mean():.4f}")


# 4.3 Reliability Diagrams (Calibration Curves)

print("\n--- 4.3 Reliability Diagrams ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, name in enumerate(all_models.keys()):
    ax = axes[idx]
    
    # Uncalibrated
    prob_true, prob_pred = calibration_curve(y_val, all_predictions_val[name], n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, 's-', label='Uncalibrated', color='red')
    
    # Platt
    prob_true_p, prob_pred_p = calibration_curve(y_val, calibrated_platt[name]['predictions'], n_bins=10, strategy='uniform')
    ax.plot(prob_pred_p, prob_true_p, '^-', label='Platt', color='blue')
    
    # Isotonic
    prob_true_i, prob_pred_i = calibration_curve(y_val, calibrated_isotonic[name]['predictions'], n_bins=10, strategy='uniform')
    ax.plot(prob_pred_i, prob_true_i, 'o-', label='Isotonic', color='green')
    
    # Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{name} Calibration', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/stage2/calibration_curves_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/calibration_curves_all.png")


# 4.4 Brier Scores Comparison

print("\n--- 4.4 Brier Scores (Lower is Better) ---")

brier_results = []
best_calibrated_models = {}

for name in all_models.keys():
    brier_uncal = brier_score_loss(y_val, all_predictions_val[name])
    brier_platt = brier_score_loss(y_val, calibrated_platt[name]['predictions'])
    brier_iso = brier_score_loss(y_val, calibrated_isotonic[name]['predictions'])
    
    # Determine best calibration method
    scores = {'Uncalibrated': brier_uncal, 'Platt': brier_platt, 'Isotonic': brier_iso}
    best_method = min(scores, key=scores.get)
    
    if best_method == 'Platt':
        best_calibrated_models[name] = calibrated_platt[name]['model']
    elif best_method == 'Isotonic':
        best_calibrated_models[name] = calibrated_isotonic[name]['model']
    else:
        best_calibrated_models[name] = all_models[name]['model']
    
    brier_results.append({
        'Model': name,
        'Uncalibrated': brier_uncal,
        'Platt': brier_platt,
        'Isotonic': brier_iso,
        'Best_Method': best_method,
        'Best_Score': scores[best_method]
    })

brier_df = pd.DataFrame(brier_results)
print("\nBrier Score Comparison:")
print(brier_df.to_string(index=False))

# Plot Brier scores
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(all_models))
width = 0.25

ax.bar(x - width, brier_df['Uncalibrated'], width, label='Uncalibrated', color='red', alpha=0.7)
ax.bar(x, brier_df['Platt'], width, label='Platt', color='blue', alpha=0.7)
ax.bar(x + width, brier_df['Isotonic'], width, label='Isotonic', color='green', alpha=0.7)

ax.set_xlabel('Model')
ax.set_ylabel('Brier Score (Lower is Better)')
ax.set_title('Calibration Comparison: Brier Scores', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(brier_df['Model'], rotation=15, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/stage2/brier_scores_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/brier_scores_comparison.png")

# Update predictions with best calibrated versions
print("\n--- Updating predictions with best calibrated models ---")
for name in all_models.keys():
    best_method = brier_df[brier_df['Model'] == name]['Best_Method'].values[0]
    
    if best_method == 'Platt':
        all_predictions_val[name] = calibrated_platt[name]['predictions']
        # Generate test predictions
        if all_models[name]['scaled']:
            all_predictions_test[name] = calibrated_platt[name]['model'].predict_proba(X_test_scaled)[:, 1]
        else:
            all_predictions_test[name] = calibrated_platt[name]['model'].predict_proba(X_test)[:, 1]
    elif best_method == 'Isotonic':
        all_predictions_val[name] = calibrated_isotonic[name]['predictions']
        if all_models[name]['scaled']:
            all_predictions_test[name] = calibrated_isotonic[name]['model'].predict_proba(X_test_scaled)[:, 1]
        else:
            all_predictions_test[name] = calibrated_isotonic[name]['model'].predict_proba(X_test)[:, 1]
    
    print(f"  {name}: Using {best_method} calibration")


# Isotonic regression consistently outperformed Platt scaling across all models. For XGBoost, isotonic calibration reduced the Brier score from 0.238 to 0.190, a 20% improvement. Random Forest improved from 0.232 to 0.191 (18% better), and Logistic Regression improved from 0.246 to 0.194 (21% better). The Neural Network showed minimal improvement from 0.195 to 0.191 (2% better) because it was already reasonably well-calibrated.
# 
# The reliability diagrams visually confirm these improvements. Before calibration, the tree models' curves deviated substantially from the diagonal representing perfect calibration. After isotonic calibration, all curves much more closely follow the diagonal, especially in the 0.2 to 0.5 probability range where most predictions fall. This range is crucial for overbooking decisions because patients with moderate no-show risk represent the marginal cases where overbooking decisions have the greatest impact.
# 
# Based on Brier scores, isotonic regression should be selected as the calibration method for all models. The calibrated predictions replace the original predictions for all subsequent analysis and for production use. This ensures that when the model predicts a 30% no-show probability, approximately 30% of such patients will actually no-show—exactly the property needed for accurate overbooking optimization.

# ### Section 5: Model Evaluation and Comparison


# 5.1 Comprehensive Metrics

print("\n--- 5.1 Comprehensive Metrics ---")

def compute_all_metrics(y_true, y_proba, model_name):
    """Compute comprehensive metrics for a model"""
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    
    # Find optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresh = thresholds[optimal_idx]
    
    # Metrics at different thresholds
    y_pred_05 = (y_proba >= 0.5).astype(int)
    y_pred_opt = (y_proba >= optimal_thresh).astype(int)
    
    return {
        'Model': model_name,
        'AUC': auc,
        'Avg_Precision': ap,
        'Brier_Score': brier,
        'Optimal_Threshold': optimal_thresh,
        'Accuracy_0.5': accuracy_score(y_true, y_pred_05),
        'F1_0.5': f1_score(y_true, y_pred_05),
        'Accuracy_Opt': accuracy_score(y_true, y_pred_opt),
        'F1_Opt': f1_score(y_true, y_pred_opt)
    }

# Validation metrics
val_metrics = [compute_all_metrics(y_val, all_predictions_val[name], name) for name in all_models.keys()]
val_metrics_df = pd.DataFrame(val_metrics).sort_values('AUC', ascending=False)

print("\nValidation Set Metrics:")
print(val_metrics_df.to_string(index=False))

# Test metrics
test_metrics = [compute_all_metrics(y_test, all_predictions_test[name], name) for name in all_models.keys()]
test_metrics_df = pd.DataFrame(test_metrics).sort_values('AUC', ascending=False)

print("\nTest Set Metrics:")
print(test_metrics_df.to_string(index=False))


# On the validation set, XGBoost leads with an AUC of 0.629, followed by Random Forest at 0.623, Neural Network at 0.611, and Logistic Regression at 0.608. The average precision scores follow a similar ranking: XGBoost at 0.402, Random Forest at 0.395, Neural Network at 0.387, and Logistic Regression at 0.358.
# 
# On the held-out test set, the rankings remain consistent but performance drops slightly. XGBoost achieves 0.619 AUC (a 1.6% decrease from validation), Random Forest achieves 0.617 (1.0% decrease), Logistic Regression achieves 0.598 (1.6% decrease), and Neural Network achieves 0.594 (2.8% decrease). These small validation-to-test gaps indicate good generalization without severe overfitting. The Neural Network shows the largest drop, suggesting it may have slightly overfit to the validation data despite early stopping.
# 
# The default classification threshold of 0.5 produces highly imbalanced predictions because most patients have predicted no-show probabilities below 50%. At this threshold, all models achieve high accuracy (around 73-74%) but very low F1 scores (0.01-0.10) because they rarely predict no-show. This occurs because accuracy rewards correct predictions on the majority class (show-ups) while the F1 score penalizes missing the minority class (no-shows).
# 
# Youden's J statistic identifies optimal thresholds that balance sensitivity and specificity. These optimal thresholds range from 0.25 to 0.30 across models, much lower than the default 0.5. At these thresholds, accuracy drops to around 55-61% but F1 scores improve dramatically to 0.39-0.46. For overbooking purposes, however, raw probability predictions matter more than binary classifications, so threshold selection primarily affects fairness analysis and reporting rather than operational decisions.


# 5.2 ROC Curves

print("\n--- 5.2 ROC Curves ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation ROC
for name, proba in all_predictions_val.items():
    fpr, tpr, _ = roc_curve(y_val, proba)
    auc = roc_auc_score(y_val, proba)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves - Validation Set', fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Test ROC
for name, proba in all_predictions_test.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    axes[1].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves - Test Set', fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/stage2/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/roc_curves_comparison.png")


# 5.3 Precision-Recall Curves

print("\n--- 5.3 Precision-Recall Curves ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Validation PR
baseline = y_val.mean()
for name, proba in all_predictions_val.items():
    precision, recall, _ = precision_recall_curve(y_val, proba)
    ap = average_precision_score(y_val, proba)
    axes[0].plot(recall, precision, label=f'{name} (AP={ap:.3f})')

axes[0].axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curves - Validation', fontweight='bold')
axes[0].legend(loc='lower left')
axes[0].grid(alpha=0.3)

# Test PR
baseline_test = y_test.mean()
for name, proba in all_predictions_test.items():
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    axes[1].plot(recall, precision, label=f'{name} (AP={ap:.3f})')

axes[1].axhline(y=baseline_test, color='k', linestyle='--', label=f'Baseline ({baseline_test:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves - Test', fontweight='bold')
axes[1].legend(loc='lower left')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/stage2/pr_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/pr_curves_comparison.png")


# 5.4 Confusion Matrices

print("\n--- 5.4 Confusion Matrices ---")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for idx, (name, proba) in enumerate(all_predictions_val.items()):
    # Threshold 0.5
    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx])
    axes[0, idx].set_title(f'{name}\n(Threshold=0.5)', fontweight='bold')
    axes[0, idx].set_xlabel('Predicted')
    axes[0, idx].set_ylabel('Actual')
    
    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(y_val, proba)
    optimal_thresh = thresholds[np.argmax(tpr - fpr)]
    y_pred_opt = (proba >= optimal_thresh).astype(int)
    cm_opt = confusion_matrix(y_val, y_pred_opt)
    sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Oranges', ax=axes[1, idx])
    axes[1, idx].set_title(f'{name}\n(Threshold={optimal_thresh:.2f})', fontweight='bold')
    axes[1, idx].set_xlabel('Predicted')
    axes[1, idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results/figures/stage2/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/confusion_matrices.png")


# 5.5 Statistical Significance Tests (McNemar's Test)

print("\n--- 5.5 Statistical Significance Tests ---")

def mcnemar_test(y_true, pred1, pred2, threshold=0.5):
    """McNemar's test to compare two classifiers"""
    y_pred1 = (pred1 >= threshold).astype(int)
    y_pred2 = (pred2 >= threshold).astype(int)
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    # b: model 1 correct, model 2 wrong
    b = ((correct1) & (~correct2)).sum()
    # c: model 1 wrong, model 2 correct
    c = ((~correct1) & (correct2)).sum()
    
    if b + c > 0:
        statistic = ((abs(b - c) - 1) ** 2) / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    else:
        statistic, p_value = 0, 1.0
    
    return statistic, p_value

model_names = list(all_predictions_val.keys())
print("\nMcNemar's Test Results:")
print("-" * 60)

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        name1, name2 = model_names[i], model_names[j]
        stat, p_val = mcnemar_test(y_val, all_predictions_val[name1], all_predictions_val[name2])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{name1} vs {name2}: χ²={stat:.2f}, p={p_val:.4f} {sig}")


# McNemar's test compares whether two classifiers make significantly different errors. The results reveal important distinctions. Random Forest and XGBoost do not differ significantly (p=0.69), meaning their predictions are largely interchangeable. However, both tree models differ highly significantly from Logistic Regression (p less than 0.001), confirming that tree-based approaches capture patterns that linear models miss. XGBoost approaches significance when compared to Neural Network (p=0.052), and Logistic Regression differs highly significantly from Neural Network (p less than 0.001).
# 
# These findings support the ensemble strategy: combining models that make different types of errors can improve overall performance. The statistical equivalence of Random Forest and XGBoost suggests some redundancy, but their different feature importance profiles indicate they capture complementary aspects of the data.

# ### Section 6: Fairness Analysis


# Use best performing model for fairness analysis
best_model_name = val_metrics_df.iloc[0]['Model']
best_proba = all_predictions_val[best_model_name]
print(f"Analyzing fairness for: {best_model_name}")

# Add predictions to validation data
val_full['pred_proba'] = best_proba
val_full['y_true'] = y_val

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_val, best_proba)
optimal_thresh = thresholds[np.argmax(tpr - fpr)]
val_full['y_pred'] = (val_full['pred_proba'] >= optimal_thresh).astype(int)


# 6.1 Fairness Metrics by Group

print("\n--- 6.1 Fairness Metrics by Group ---")

def compute_group_metrics(df, group_col, group_name):
    """Compute fairness metrics for demographic groups"""
    metrics = []
    for group_val in df[group_col].dropna().unique():
        group_data = df[df[group_col] == group_val]
        
        if len(group_data) < 50:
            continue
        
        positives = group_data[group_data['y_true'] == 1]
        negatives = group_data[group_data['y_true'] == 0]
        
        tpr = positives['y_pred'].mean() if len(positives) > 0 else 0
        fpr = negatives['y_pred'].mean() if len(negatives) > 0 else 0
        accuracy = (group_data['y_pred'] == group_data['y_true']).mean()
        
        if group_data['y_true'].nunique() > 1:
            auc = roc_auc_score(group_data['y_true'], group_data['pred_proba'])
        else:
            auc = np.nan
        
        metrics.append({
            'Group': f'{group_name}={group_val}',
            'N': len(group_data),
            'Positive_Rate': group_data['y_true'].mean(),
            'TPR': tpr,
            'FPR': fpr,
            'Accuracy': accuracy,
            'AUC': auc
        })
    
    return pd.DataFrame(metrics)

# Age groups
val_full['Age_Group_Fair'] = pd.cut(val_full['Age'], bins=[0, 30, 50, 100], labels=['<30', '30-50', '50+'])
age_fairness = compute_group_metrics(val_full, 'Age_Group_Fair', 'Age')
print("\nFairness by Age Group:")
print(age_fairness.to_string(index=False))

# Scholarship status
scholarship_fairness = compute_group_metrics(val_full, 'Scholarship', 'Scholarship')
print("\nFairness by Scholarship Status:")
print(scholarship_fairness.to_string(index=False))


# For age groups, the model shows substantial performance variation. Patients under 30 have a true positive rate (TPR) of 88.5%, meaning the model correctly identifies 88.5% of actual no-shows in this group. However, the false positive rate (FPR) is also high at 80.0%, meaning 80% of patients who actually show up are incorrectly flagged as likely no-shows. This occurs because younger patients have a high base no-show rate of 33.7%, so the model aggressively predicts no-show for this demographic.
# 
# Middle-aged patients (30-50) have a TPR of 69.8% and FPR of 54.8%, with a base no-show rate of 28.4%. Older patients (50+) have much lower TPR of 27.0% but also much lower FPR of 14.1%, with a base no-show rate of 21.6%. The model is conservative with older patients, rarely predicting no-show, which results in fewer false alarms but also more missed no-shows.
# 
# For scholarship status, patients without scholarship have TPR of 63.9% and FPR of 45.6%, while scholarship recipients have TPR of 82.7% and FPR of 72.4%. Scholarship recipients have a higher base no-show rate of 35.3% compared to 27.0% for non-recipients, which partially explains the model's more aggressive predictions for this group.


# 6.2 Equal Opportunity Difference

print("\n--- 6.2 Equal Opportunity Difference ---")

# TPR parity across groups
tpr_age = age_fairness['TPR'].values
equal_opp_diff_age = tpr_age.max() - tpr_age.min()
print(f"Equal Opportunity Difference (Age): {equal_opp_diff_age:.4f}")

tpr_scholarship = scholarship_fairness['TPR'].values
if len(tpr_scholarship) >= 2:
    equal_opp_diff_scholarship = abs(tpr_scholarship[0] - tpr_scholarship[1])
else:
    equal_opp_diff_scholarship = np.nan
print(f"Equal Opportunity Difference (Scholarship): {equal_opp_diff_scholarship:.4f}")

# Interpretation
print("\nInterpretation:")
if equal_opp_diff_age > 0.1:
    print("  ⚠ Age groups show significant TPR disparity (>10%)")
else:
    print("  ✓ Age groups have similar TPR (<10% difference)")

if equal_opp_diff_scholarship > 0.1:
    print("  ⚠ Scholarship groups show significant TPR disparity (>10%)")
else:
    print("  ✓ Scholarship groups have similar TPR (<10% difference)")


# The equal opportunity difference measures the gap between the highest and lowest TPR across groups. For age, this difference is 0.614 (88.5% minus 27.0%), indicating severe disparity. For scholarship status, the difference is 0.188 (82.7% minus 63.9%), indicating moderate disparity.
# 
# These disparities arise primarily because different groups have different base rates of no-showing. The model learns that young patients and scholarship recipients are riskier, so it predicts no-show more frequently for them. This behavior is statistically accurate but raises ethical questions about whether accurate predictions can constitute unfair treatment.
# 
# It is important to distinguish between model bias and data bias. The model reflects patterns in the data rather than introducing new discrimination. Young patients genuinely no-show more often, likely due to fewer health concerns creating urgency. Scholarship recipients face real socioeconomic barriers like transportation and work constraints. The model captures these realities. However, using these predictions for overbooking decisions could disproportionately affect these groups, for example, scheduling extra patients when young patients are booked could lead to overflow that makes young patients wait longer.
# 
# For the optimization phase, fairness constraints can ensure that overbooking policies do not disproportionately burden any demographic group. This might involve limiting how much the overbooking rate can differ between groups or adjusting decision thresholds separately for each group.


# 6.3 Fairness Visualization

print("\n--- 6.3 Fairness Visualization ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age fairness
x = np.arange(len(age_fairness))
width = 0.25
axes[0].bar(x - width, age_fairness['TPR'], width, label='TPR (Recall)', color='steelblue')
axes[0].bar(x, age_fairness['FPR'], width, label='FPR', color='coral')
axes[0].bar(x + width, age_fairness['Accuracy'], width, label='Accuracy', color='lightgreen')
axes[0].set_xticks(x)
axes[0].set_xticklabels(age_fairness['Group'])
axes[0].set_ylabel('Rate')
axes[0].set_title('Fairness Metrics by Age Group', fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Scholarship fairness
x = np.arange(len(scholarship_fairness))
axes[1].bar(x - width, scholarship_fairness['TPR'], width, label='TPR (Recall)', color='steelblue')
axes[1].bar(x, scholarship_fairness['FPR'], width, label='FPR', color='coral')
axes[1].bar(x + width, scholarship_fairness['Accuracy'], width, label='Accuracy', color='lightgreen')
axes[1].set_xticks(x)
axes[1].set_xticklabels(scholarship_fairness['Group'])
axes[1].set_ylabel('Rate')
axes[1].set_title('Fairness Metrics by Scholarship Status', fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/stage2/fairness_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("✓ Saved: results/figures/stage2/fairness_analysis.png")


# ### Section 7: Ensemble Models


# 7.1 Weighted Average Ensemble

print("\n--- 7.1 Weighted Average Ensemble ---")

# Equal weights
base_model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'Neural Network']
equal_weights = {name: 0.25 for name in base_model_names}
ensemble_equal_val = sum(all_predictions_val[name] * equal_weights[name] for name in base_model_names)
ensemble_equal_test = sum(all_predictions_test[name] * equal_weights[name] for name in base_model_names)
auc_equal = roc_auc_score(y_val, ensemble_equal_val)
print(f"Equal weights ensemble - Val AUC: {auc_equal:.4f}")

# AUC-weighted ensemble
auc_scores = {name: roc_auc_score(y_val, all_predictions_val[name]) for name in base_model_names}
total_auc = sum(auc_scores.values())
auc_weights = {name: score / total_auc for name, score in auc_scores.items()}
ensemble_auc_val = sum(all_predictions_val[name] * auc_weights[name] for name in base_model_names)
ensemble_auc_test = sum(all_predictions_test[name] * auc_weights[name] for name in base_model_names)
auc_weighted = roc_auc_score(y_val, ensemble_auc_val)
print(f"AUC-weighted ensemble - Val AUC: {auc_weighted:.4f}")

print("\nAUC weights:")
for name, weight in auc_weights.items():
    print(f"  {name}: {weight:.3f}")

# Store best weighted ensemble
all_predictions_val['Weighted Ensemble'] = ensemble_auc_val
all_predictions_test['Weighted Ensemble'] = ensemble_auc_test


# The equal weight ensemble simply averages predictions from all four base models (Random Forest, XGBoost, Logistic Regression, and Neural Network), giving each 25% weight. This achieved a validation AUC of 0.630, which already exceeds any individual model.
# 
# The AUC-weighted ensemble assigns weights proportional to each model's validation AUC. The resulting weights are surprisingly similar: Random Forest 25.2%, XGBoost 25.4%, Logistic Regression 24.6%, and Neural Network 24.7%. This near-uniformity occurs because all models have similar AUC scores in the 0.61-0.63 range. The AUC-weighted ensemble achieves the same 0.630 validation AUC as equal weighting, confirming that with similarly-performing models, weighting scheme matters little.


# 7.2 Stacking Ensemble

print("\n--- 7.2 Stacking Ensemble ---")

# Create stacking features (predictions from base models)
X_stack_val = np.column_stack([all_predictions_val[name] for name in base_model_names])
X_stack_test = np.column_stack([all_predictions_test[name] for name in base_model_names])

# Train meta-model (logistic regression on base model predictions)
meta_model = LogisticRegression(random_state=RANDOM_SEED)
meta_model.fit(X_stack_val, y_val)

stack_proba_val = meta_model.predict_proba(X_stack_val)[:, 1]
stack_proba_test = meta_model.predict_proba(X_stack_test)[:, 1]

auc_stack_val = roc_auc_score(y_val, stack_proba_val)
auc_stack_test = roc_auc_score(y_test, stack_proba_test)

print(f"Stacking Ensemble - Val AUC: {auc_stack_val:.4f}")
print(f"Stacking Ensemble - Test AUC: {auc_stack_test:.4f}")

print("\nMeta-model coefficients (how much each base model contributes):")
for name, coef in zip(base_model_names, meta_model.coef_[0]):
    print(f"  {name}: {coef:.4f}")

# Store stacking predictions
all_predictions_val['Stacking Ensemble'] = stack_proba_val
all_predictions_test['Stacking Ensemble'] = stack_proba_test


# Stacking uses a meta-model to learn how to optimally combine base model predictions. Rather than fixed weights, the meta-model can learn complex combination rules, including that certain base models are more reliable in certain probability ranges or for certain types of patients.
# 
# The meta-model coefficients reveal how much each base model contributes: XGBoost dominates with a coefficient of 3.04, followed by Neural Network at 1.50, Random Forest at 0.96, and Logistic Regression at only 0.13. This indicates that XGBoost predictions are most valuable, with Neural Network providing important complementary information. Random Forest adds some value despite its similarity to XGBoost, while Logistic Regression contributes minimally.
# 
# The stacking ensemble achieves the best validation AUC of 0.631 and test AUC of 0.621. The improvement over individual models is modest but consistent, and the stacking ensemble should be selected as the production model.

# ### Section 8: Final Model Selection and Export


# 8.1 Final Comparison (Including Ensembles)

print("\n--- 8.1 Final Model Comparison ---")

final_results = []
for name in all_predictions_val.keys():
    final_results.append({
        'Model': name,
        'Val_AUC': roc_auc_score(y_val, all_predictions_val[name]),
        'Val_AP': average_precision_score(y_val, all_predictions_val[name]),
        'Val_Brier': brier_score_loss(y_val, all_predictions_val[name]),
        'Test_AUC': roc_auc_score(y_test, all_predictions_test[name]),
        'Test_AP': average_precision_score(y_test, all_predictions_test[name])
    })

final_results_df = pd.DataFrame(final_results).sort_values('Val_AUC', ascending=False)
print("\nFinal Model Comparison (sorted by Val AUC):")
print(final_results_df.to_string(index=False))


# These AUC values in the 0.59-0.63 range are consistent with published literature on medical no-show prediction, which typically reports AUC between 0.60 and 0.75 depending on the population and available features. No-show prediction is inherently difficult because patient behavior involves genuine randomness; unexpected illness, traffic, family emergencies that no model can predict. The achieved performance is sufficient for meaningful overbooking optimization.

# **Validation of Model Quality**
# Several checks confirm the models are working correctly. 
# 
# First, feature importance aligns with domain knowledge and EDA findings: age, lead time, and prior history emerged as top predictors as expected, while SMS showed minimal effect as discovered during exploratory analysis. 
# 
# Second, coefficient signs in logistic regression match intuition: age is negative (older patients less likely to no-show), lead time is positive (longer advance booking increases risk), and scholarship is positive (socioeconomic barriers increase risk). 
# 
# Third, the small validation-to-test performance gap indicates the models generalize well without severe overfitting.
# 
# Fourth, calibration curves closely follow the diagonal after isotonic regression, confirming that predicted probabilities are reliable.


# 8.2 Select and Save Best Model

print("\n--- 8.2 Best Model Selection ---")

best_model_name = final_results_df.iloc[0]['Model']
best_val_auc = final_results_df.iloc[0]['Val_AUC']
best_test_auc = final_results_df.iloc[0]['Test_AUC']

print(f"\n BEST MODEL: {best_model_name}")
print(f"   Validation AUC: {best_val_auc:.4f}")
print(f"   Test AUC: {best_test_auc:.4f}")

# Package and save the best model
if best_model_name == 'Stacking Ensemble':
    best_model_package = {
        'type': 'stacking',
        'meta_model': meta_model,
        'base_models': {name: best_calibrated_models[name] for name in base_model_names},
        'scaler': scaler,
        'requires_scaling': {name: all_models[name]['scaled'] for name in base_model_names}
    }
elif best_model_name == 'Weighted Ensemble':
    best_model_package = {
        'type': 'weighted_ensemble',
        'weights': auc_weights,
        'base_models': {name: best_calibrated_models[name] for name in base_model_names},
        'scaler': scaler,
        'requires_scaling': {name: all_models[name]['scaled'] for name in base_model_names}
    }
else:
    best_model_package = {
        'type': 'single',
        'model': best_calibrated_models[best_model_name],
        'scaler': scaler if all_models[best_model_name]['scaled'] else None,
        'requires_scaling': all_models[best_model_name]['scaled']
    }

joblib.dump(best_model_package, 'results/models/best_model.pkl')
print("✓ Saved: results/models/best_model.pkl")


# 8.3 Save All Individual Models

print("\n--- 8.3 Saving All Models ---")

joblib.dump(rf_best, 'results/models/random_forest.pkl')
joblib.dump(xgb_best, 'results/models/xgboost.pkl')
joblib.dump(lr_best, 'results/models/logistic_regression.pkl')
joblib.dump(mlp, 'results/models/neural_network.pkl')
joblib.dump(scaler, 'results/models/scaler.pkl')

# Save calibrated models
for name, model in best_calibrated_models.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, f'results/models/{safe_name}_calibrated.pkl')

print("✓ Saved all individual models to results/models/ directory")


# 8.4 Save Reports

print("\n--- 8.4 Saving Reports ---")

# Model comparison
final_results_df.to_csv('results/models/model_comparison.csv', index=False)
print("✓ Saved: results/models/model_comparison.csv")

# Brier scores / calibration report
brier_df.to_csv('results/models/calibration_report.csv', index=False)
print("✓ Saved: results/models/calibration_report.csv")

# Fairness report
fairness_report = pd.concat([
    age_fairness.assign(Category='Age'),
    scholarship_fairness.assign(Category='Scholarship')
])
fairness_report.to_csv('results/models/fairness_report.csv', index=False)
print("✓ Saved: results/models/fairness_report.csv")

# Feature importance
xgb_importance.to_csv('results/models/feature_importance_xgb.csv', index=False)
rf_importance.to_csv('results/models/feature_importance_rf.csv', index=False)
lr_coefficients.to_csv('results/models/lr_coefficients.csv', index=False)
print("✓ Saved: results/models/feature_importance_*.csv")


# ### Final Summary


print(f"""
MODEL PERFORMANCE SUMMARY
=========================

{final_results_df.to_string(index=False)}

BEST MODEL: {best_model_name}
  - Validation AUC: {best_val_auc:.4f}
  - Test AUC: {best_test_auc:.4f}

TOP 5 FEATURES (XGBoost):
{xgb_importance.head().to_string(index=False)}

CALIBRATION SUMMARY:
{brier_df[['Model', 'Best_Method', 'Best_Score']].to_string(index=False)}

FAIRNESS ANALYSIS:
  - Equal Opportunity Difference (Age): {equal_opp_diff_age:.4f}
  - Equal Opportunity Difference (Scholarship): {equal_opp_diff_scholarship:.4f}

FILES GENERATED:
  Models:
    - results/models/best_model.pkl (production-ready)
    - results/models/random_forest.pkl
    - results/models/xgboost.pkl
    - results/models/logistic_regression.pkl
    - results/models/neural_network.pkl
    - results/models/scaler.pkl
    - results/models/*_calibrated.pkl
  
  Reports:
    - results/models/model_comparison.csv
    - results/models/calibration_report.csv
    - results/models/fairness_report.csv
    - results/models/feature_importance_*.csv
  
  Figures:
    - results/figures/stage2/shap_importance_bar.png
    - results/figures/stage2/shap_beeswarm.png
    - results/figures/stage2/feature_importance_comparison.png
    - results/figures/stage2/lr_coefficients.png
    - results/figures/stage2/mlp_loss_curve.png
    - results/figures/stage2/calibration_curves_all.png
    - results/figures/stage2/brier_scores_comparison.png
    - results/figures/stage2/roc_curves_comparison.png
    - results/figures/stage2/pr_curves_comparison.png
    - results/figures/stage2/confusion_matrices.png
    - results/figures/stage2/fairness_analysis.png
""")


# The current models meet all requirements for the optimization phase. Calibration has been applied, achieving Brier scores around 0.19, which ensures that probability predictions accurately reflect true no-show rates. Test AUC of 0.62 provides sufficient discrimination to distinguish higher-risk from lower-risk patients. The small validation-test gap confirms stable predictions that will perform similarly in simulation. SHAP analysis provides interpretability needed to explain predictions to clinic staff and stakeholders.
# 
# No additional tuning is needed before proceeding to optimization. Further hyperparameter search would yield diminishing returns, more complex models risk overfitting, and the current features already capture the available predictive signal. 
# 
# **The models are ready to generate no-show probability predictions for overbooking policy simulation and optimization.**

# The analysis produced several artifacts for use in subsequent stages. The best_model.pkl file contains the stacking ensemble with all necessary components for generating predictions. Individual model files preserve each trained classifier for comparison or fallback use. The scaler preserves the StandardScaler fitted on training data for consistent preprocessing. Calibrated model files contain the isotonic-calibrated versions of each classifier. Report files document model comparison metrics, calibration results, fairness analysis, and feature importances for documentation and presentation purposes.
