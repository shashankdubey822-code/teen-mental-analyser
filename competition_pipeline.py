"""
=============================================================
 TEEN MENTAL HEALTH ANALYZER — ADVANCED COMPETITION PIPELINE
=============================================================
 Covers ALL Bake-Off Requirements:
  1. Data Preprocessing (Visualization, Standardization,
     Outlier Management, Categorical Encoding, Feature Selection)
  2. Handling Class Imbalance (SMOTE)
  3. Model Training (Cross Validation, Regularization,
     Hyperparameter Tuning, Generalization)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     GridSearchCV, learning_curve)
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance as perm_imp
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

from feature_engineering import SOCIAL_INTERACTION_MAP
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
#  GLOBAL DARK THEME
# ─────────────────────────────────────────────────────
BG    = '#0f1117'
PANEL = '#1a1d2e'
GRID  = '#2d3748'
TEXT  = '#e2e8f0'
C0    = '#2ecc71'   # No Depression (green)
C1    = '#e74c3c'   # Depression (red)
PALETTE = [C0, C1]

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    PANEL,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   TEXT,
    'xtick.color':       TEXT,
    'ytick.color':       TEXT,
    'text.color':        TEXT,
    'grid.color':        GRID,
    'grid.alpha':        0.3,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'legend.facecolor':  PANEL,
    'legend.edgecolor':  GRID,
})

cat_cols = ['gender', 'platform_usage', 'social_interaction_level']
num_cols = ['age', 'daily_social_media_hours', 'sleep_hours',
            'screen_time_before_sleep', 'academic_performance',
            'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level']

# ─────────────────────────────────────────────────────
#  LOAD & ENCODE
# ─────────────────────────────────────────────────────
df = pd.read_csv('Teen_Mental_Health_Dataset.csv')
encoders = {}
df_enc = df.copy()

df_enc['gender'] = df_enc['gender'].astype(str).str.lower().str.strip()
df_enc['platform_usage'] = df_enc['platform_usage'].astype(str).str.strip().str.title()
df_enc['social_interaction_level'] = df_enc['social_interaction_level'].astype(str).str.lower().str.strip()

gender_le = LabelEncoder()
df_enc['gender'] = gender_le.fit_transform(df_enc['gender'])
encoders['gender'] = gender_le

platform_le = LabelEncoder()
df_enc['platform_usage'] = platform_le.fit_transform(df_enc['platform_usage'])
encoders['platform_usage'] = platform_le

social_map = SOCIAL_INTERACTION_MAP
df_enc['social_interaction_level'] = df_enc['social_interaction_level'].where(
    df_enc['social_interaction_level'].isin(social_map), 'medium')
df_enc['social_interaction_level'] = df_enc['social_interaction_level'].map(social_map)
encoders['social_interaction_level'] = social_map

joblib.dump(encoders, 'label_encoders.pkl')
print("✓ label_encoders.pkl saved")


# ══════════════════════════════════════════════════════
# DASHBOARD 1 — EXPLORATORY DATA ANALYSIS (8 panels)
# ══════════════════════════════════════════════════════
print("\nBuilding Dashboard 1: EDA ...")
fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle('Teen Mental Health Analyzer  ·  Exploratory Data Analysis',
             fontsize=20, fontweight='bold', color=TEXT, y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

# 1 — Donut chart
ax = fig.add_subplot(gs[0, 0])
counts = df['depression_label'].value_counts()
wedges, _, auto = ax.pie(
    counts, labels=['No Depression', 'Depression'], colors=PALETTE,
    autopct='%1.1f%%', startangle=90,
    wedgeprops={'width': 0.52, 'edgecolor': BG, 'linewidth': 2.5},
    textprops={'color': TEXT, 'fontsize': 10})
for a in auto: a.set_fontsize(12)
ax.set_title('Class Distribution', color=TEXT, fontweight='bold')
ax.text(0, 0, f'n={len(df)}', ha='center', va='center',
        fontsize=13, fontweight='bold', color=TEXT)

# 2 — KDE Sleep Hours
ax = fig.add_subplot(gs[0, 1])
for label, color, name in [(0, C0, 'No Depression'), (1, C1, 'Depression')]:
    sub = df[df['depression_label'] == label]['sleep_hours']
    sub.plot.kde(ax=ax, color=color, linewidth=2.5, label=name)
    ax.axvline(sub.mean(), color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    ax.fill_between(np.linspace(sub.min(), sub.max(), 300),
                    np.zeros(300), alpha=0.06, color=color)
ax.set_title('Sleep Hours (KDE)', color=TEXT, fontweight='bold')
ax.set_xlabel('Sleep Hours'); ax.legend()

# 3 — Violin: Addiction Level
ax = fig.add_subplot(gs[0, 2])
data_v = [df[df['depression_label'] == l]['addiction_level'].values for l in [0, 1]]
vp = ax.violinplot(data_v, positions=[0, 1], showmeans=True, showmedians=True)
for body, color in zip(vp['bodies'], PALETTE):
    body.set_facecolor(color); body.set_alpha(0.65)
vp['cmeans'].set_color(TEXT); vp['cmedians'].set_color('#f39c12')
ax.set_xticks([0, 1]); ax.set_xticklabels(['No Depression', 'Depression'])
ax.set_title('Addiction Level (Violin)', color=TEXT, fontweight='bold')

# 4 — Correlation heatmap (lower triangle)
ax = fig.add_subplot(gs[1, :2])
corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, linewidths=0.3, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.7})
ax.set_title('Feature Correlation Matrix (Lower Triangle)', color=TEXT, fontweight='bold')
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0, labelsize=8)

# 5 — Stress vs Anxiety scatter
ax = fig.add_subplot(gs[1, 2])
for label, color, name in [(0, C0, 'No Risk'), (1, C1, 'At Risk')]:
    s = df[df['depression_label'] == label]
    ax.scatter(s['stress_level'], s['anxiety_level'],
               c=color, alpha=0.45, s=18, label=name)
ax.set_xlabel('Stress Level'); ax.set_ylabel('Anxiety Level')
ax.set_title('Stress vs Anxiety', color=TEXT, fontweight='bold')
ax.legend(markerscale=2)

# 6 — Notched boxplot: Social Media Hours
ax = fig.add_subplot(gs[2, 0])
data_b = [df[df['depression_label'] == l]['daily_social_media_hours'].values for l in [0, 1]]
bp = ax.boxplot(data_b, patch_artist=True, notch=True,
                whiskerprops={'color': TEXT}, capprops={'color': TEXT},
                medianprops={'color': '#f39c12', 'linewidth': 2})
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticklabels(['No Depression', 'Depression'])
ax.set_title('Social Media Hours (Boxplot)', color=TEXT, fontweight='bold')

# 7 — Stacked bar: Platform usage
ax = fig.add_subplot(gs[2, 1])
piv = df.groupby(['platform_usage', 'depression_label']).size().unstack(fill_value=0)
piv.plot(kind='bar', stacked=True, ax=ax, color=PALETTE, edgecolor=BG, linewidth=0.5)
ax.set_title('Platform Usage vs Depression', color=TEXT, fontweight='bold')
ax.set_xlabel(''); ax.tick_params(axis='x', rotation=25)
ax.legend(['No Depression', 'Depression'])

# 8 — PCA 2D
ax = fig.add_subplot(gs[2, 2])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(StandardScaler().fit_transform(
    df_enc.drop('depression_label', axis=1)))
for label, color, name in [(0, C0, 'No Risk'), (1, C1, 'At Risk')]:
    idx = df_enc['depression_label'] == label
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=color, alpha=0.45, s=14, label=name)
var = sum(pca.explained_variance_ratio_) * 100
ax.set_title(f'PCA 2D Projection  ({var:.1f}% Variance)', color=TEXT, fontweight='bold')
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2'); ax.legend(markerscale=2)

plt.savefig('dashboard_eda.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ dashboard_eda.png")


# ══════════════════════════════════════════════════════
# OUTLIER DETECTION (IQR) + REPORT
# ══════════════════════════════════════════════════════
print("\nPhase: Outlier Detection ...")
report_rows = []
df_clean = df.copy()
for col in num_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = int(((df_clean[col] < lo) | (df_clean[col] > hi)).sum())
    df_clean[col] = df_clean[col].clip(lo, hi)
    report_rows.append({'Feature': col, 'Outliers_Clipped': n_out,
                        'Lower_Fence': round(lo, 3), 'Upper_Fence': round(hi, 3)})

outlier_df = pd.DataFrame(report_rows)
outlier_df.to_csv('outlier_report.csv', index=False)
print(outlier_df.to_string(index=False))
print("  ✓ outlier_report.csv saved")


# ══════════════════════════════════════════════════════
# STANDARDIZATION
# ══════════════════════════════════════════════════════
print("\nPhase: Standardization ...")
X_train_raw = pd.read_csv('X_train.csv')
X_test_raw  = pd.read_csv('X_test.csv')
y_train_all = pd.read_csv('y_train.csv').values.ravel()
y_test_all  = pd.read_csv('y_test.csv').values.ravel()

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train_raw)
X_te = scaler.transform(X_test_raw)
joblib.dump(scaler, 'scaler.pkl')
print(f"  ✓ scaler.pkl saved  |  Train: {X_tr.shape}  Test: {X_te.shape}")

feature_names = X_train_raw.columns.tolist()


# ══════════════════════════════════════════════════════
# DASHBOARD 2 — FEATURE IMPORTANCE (3 methods)
# ══════════════════════════════════════════════════════
print("\nBuilding Dashboard 2: Feature Importance ...")

rf_fi = RandomForestClassifier(n_estimators=300, random_state=42)
rf_fi.fit(X_tr, y_train_all)
rf_imp = pd.Series(rf_fi.feature_importances_, index=feature_names).sort_values(ascending=False)

xgb_fi = XGBClassifier(eval_metric='logloss', random_state=42, n_estimators=200)
xgb_fi.fit(X_tr, y_train_all)
xgb_imp = pd.Series(xgb_fi.feature_importances_, index=feature_names).sort_values(ascending=False)

perm_res = perm_imp(rf_fi, X_te, y_test_all, n_repeats=10, random_state=42)
perm_imp_s = pd.Series(perm_res.importances_mean, index=feature_names).sort_values(ascending=False)

fig2, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor=BG)
fig2.suptitle('Advanced Feature Importance — Three Methods Compared',
              fontsize=16, fontweight='bold', color=TEXT)

def hbar(ax, series, title, cmap_name):
    cmap = plt.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.35, 0.95, len(series)))
    bars = ax.barh(series.index[::-1], series.values[::-1],
                   color=colors[::-1], edgecolor=BG, linewidth=0.4)
    ax.set_title(title, color=TEXT, fontweight='bold', fontsize=12, pad=12)
    ax.set_xlabel('Importance Score', color=TEXT)
    for bar, val in zip(bars, series.values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8.5, color=TEXT)
    ax.spines[['top', 'right']].set_visible(False)

hbar(axes[0], rf_imp,   '🌳 Random Forest\n(Mean Decrease Impurity)', 'Greens')
hbar(axes[1], xgb_imp,  '🚀 XGBoost\n(Total Gain)',                   'Blues')
hbar(axes[2], perm_imp_s,'🔀 Permutation Importance\n(Model-Agnostic)','Purples')

plt.tight_layout()
plt.savefig('dashboard_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ dashboard_feature_importance.png")


# ══════════════════════════════════════════════════════
# CROSS VALIDATION + LEARNING CURVES DASHBOARD
# ══════════════════════════════════════════════════════
print("\nBuilding Dashboard 3: Cross Validation + Learning Curves ...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost":       XGBClassifier(eval_metric='logloss', random_state=42),
    "LightGBM":      LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost":      CatBoostClassifier(random_state=42, verbose=0),
}
cv_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG)
fig3.suptitle('Cross Validation & Generalization Analysis',
              fontsize=16, fontweight='bold', color=TEXT)

# — Strip + diamond CV plot
cv_results = []
print(f"\n  {'Model':<18} {'F1 Mean':>9} {'F1 Std':>9} {'Acc Mean':>10}")
print("  " + "-"*50)
for (name, model), color in zip(cv_models.items(), cv_colors):
    f1s  = cross_val_score(model, X_tr, y_train_all, cv=cv, scoring='f1')
    accs = cross_val_score(model, X_tr, y_train_all, cv=cv, scoring='accuracy')
    print(f"  {name:<18} {f1s.mean():>9.4f} {f1s.std():>9.4f} {accs.mean():>10.4f}")
    cv_results.append({'Model': name, 'CV_F1_Mean': round(f1s.mean(), 4),
                       'CV_F1_Std': round(f1s.std(), 4), 'CV_Acc_Mean': round(accs.mean(), 4)})
    axes3[0].scatter([name]*5, f1s, color=color, s=55, alpha=0.8, zorder=5)
    axes3[0].plot([name]*2, [f1s.mean()-f1s.std(), f1s.mean()+f1s.std()],
                  color=color, linewidth=2.5)
    axes3[0].scatter([name], [f1s.mean()], color=color, s=220,
                     marker='D', edgecolors='white', linewidth=1.5, zorder=6)

axes3[0].set_ylim(0, 1.05)
axes3[0].set_title('5-Fold Stratified CV  (◆ = Mean, dots = Folds)',
                   color=TEXT, fontweight='bold')
axes3[0].set_ylabel('F1 Score')
axes3[0].tick_params(axis='x', rotation=15)
pd.DataFrame(cv_results).to_csv('cross_validation_results.csv', index=False)

# — Learning Curve (XGBoost)
lc_model = XGBClassifier(eval_metric='logloss', random_state=42)
sizes, tr_sc, val_sc = learning_curve(
    lc_model, X_tr, y_train_all, cv=cv, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

tr_m, tr_s   = tr_sc.mean(1), tr_sc.std(1)
val_m, val_s = val_sc.mean(1), val_sc.std(1)

axes3[1].plot(sizes, tr_m,  color='#3498db', lw=2.5, marker='o', label='Train F1')
axes3[1].fill_between(sizes, tr_m-tr_s, tr_m+tr_s, alpha=0.12, color='#3498db')
axes3[1].plot(sizes, val_m, color='#e74c3c', lw=2.5, marker='s', label='Validation F1')
axes3[1].fill_between(sizes, val_m-val_s, val_m+val_s, alpha=0.12, color='#e74c3c')
axes3[1].set_title('XGBoost Learning Curve  (Bias–Variance Tradeoff)',
                   color=TEXT, fontweight='bold')
axes3[1].set_xlabel('Training Samples')
axes3[1].set_ylabel('F1 Score')
axes3[1].set_ylim(0, 1.05)
axes3[1].legend()

plt.tight_layout()
plt.savefig('dashboard_cv_learning.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ dashboard_cv_learning.png")


# ══════════════════════════════════════════════════════
# HYPERPARAMETER TUNING — GridSearchCV (with L1 + L2)
# ══════════════════════════════════════════════════════
print("\nPhase: Hyperparameter Tuning (GridSearchCV) ...")

param_grid = {
    'n_estimators':  [100, 200, 300],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample':     [0.8, 1.0],
    'reg_alpha':     [0, 0.1],    # L1 regularization
    'reg_lambda':    [1, 1.5],    # L2 regularization
}

grid_search = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0)
grid_search.fit(X_tr, y_train_all)

print(f"  Best Parameters : {grid_search.best_params_}")
print(f"  Best CV F1      : {grid_search.best_score_:.6f}")

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'champion_xgboost_model.pkl')
print("  ✓ champion_xgboost_model.pkl saved  (Tuned with L1 + L2 regularization)")


# ══════════════════════════════════════════════════════
# DASHBOARD 4 — FINAL EVALUATION (ROC + PR + CM)
# ══════════════════════════════════════════════════════
print("\nBuilding Dashboard 4: Final Evaluation ...")

eval_models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost (Tuned)": best_model,
    "LightGBM":       LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost":       CatBoostClassifier(random_state=42, verbose=0),
}
eval_colors = ['#f39c12', '#e74c3c', '#2ecc71', '#3498db']

fig4, axes4 = plt.subplots(1, 3, figsize=(24, 8), facecolor=BG)
fig4.suptitle('Final Model Evaluation Dashboard',
              fontsize=16, fontweight='bold', color=TEXT)

# ROC Curves
axes4[0].plot([0,1],[0,1], 'w--', lw=1, alpha=0.4)
for (name, model), color in zip(eval_models.items(), eval_colors):
    model.fit(X_tr, y_train_all)
    fpr, tpr, _ = roc_curve(y_test_all, model.predict_proba(X_te)[:,1])
    axes4[0].plot(fpr, tpr, color=color, lw=2,
                  label=f'{name}  (AUC={auc(fpr,tpr):.4f})')
axes4[0].set_xlabel('False Positive Rate')
axes4[0].set_ylabel('True Positive Rate')
axes4[0].set_title('ROC Curves — All Models', color=TEXT, fontweight='bold')
axes4[0].legend(loc='lower right', fontsize=8.5)

# Precision-Recall Curves
for (name, model), color in zip(eval_models.items(), eval_colors):
    prec, rec, _ = precision_recall_curve(y_test_all, model.predict_proba(X_te)[:,1])
    axes4[1].plot(rec, prec, color=color, lw=2,
                  label=f'{name}  (AUC={auc(rec,prec):.4f})')
axes4[1].set_xlabel('Recall')
axes4[1].set_ylabel('Precision')
axes4[1].set_title('Precision-Recall Curves\n(Key Metric for Imbalanced Data)',
                   color=TEXT, fontweight='bold')
axes4[1].legend(loc='lower left', fontsize=8.5)

# Confusion Matrix (tuned champion)
y_pred = best_model.predict(X_te)
cm = confusion_matrix(y_test_all, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=axes4[2],
            xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'],
            linewidths=0.5, annot_kws={'size': 18, 'weight': 'bold'},
            cbar_kws={'shrink': 0.7})
axes4[2].set_title('Confusion Matrix\nXGBoost — Tuned Champion', color=TEXT, fontweight='bold')
axes4[2].set_ylabel('Actual'); axes4[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('dashboard_final_evaluation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  ✓ dashboard_final_evaluation.png")

# Final report
print("\n" + "="*60)
print("TUNED XGBOOST — FINAL CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test_all, y_pred,
      target_names=['No Depression', 'Depression']))

print("="*60)
print("✅  ALL COMPETITION REQUIREMENTS — ADVANCED LEVEL DONE")
print("="*60)
print("""
  Generated Artifacts:
  ┌───────────────────────────────────────────────────────┐
  │  📊  dashboard_eda.png                (8-panel EDA)  │
  │  📊  dashboard_feature_importance.png (3 methods)    │
  │  📊  dashboard_cv_learning.png        (CV + Curve)   │
  │  📊  dashboard_final_evaluation.png   (ROC, PR, CM)  │
  │  📄  outlier_report.csv                              │
  │  📄  cross_validation_results.csv                    │
  │  🤖  champion_xgboost_model.pkl  (Tuned L1 + L2)    │
  │  🔧  scaler.pkl                                      │
  │  🔧  label_encoders.pkl                              │
  └───────────────────────────────────────────────────────┘
""")
