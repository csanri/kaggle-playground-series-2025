import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error

from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv("test.csv")
idx = test_df["id"]
test_df = test_df.drop("id", axis=1)

target = "Price"
random_state = 42
opt_iter = 100

train_df.columns = train_df.columns.str.replace(" ", "_")
test_df.columns = test_df.columns.str.replace(" ", "_")

print(train_df.info())
print("=" * 30)
print(train_df.describe())
print("=" * 30)

cols = train_df.columns.tolist()

for col in cols:
    print("-" * 30)
    print(train_df[col].value_counts())

print("=" * 30)
print(train_df.isna().sum())
print("=" * 30)

cat_cols = train_df.select_dtypes("object").columns.tolist()
num_cols = train_df.select_dtypes(exclude="object").columns.tolist()
num_cols.remove(target)

colors = sns.color_palette("tab10")

# fig, axes = plt.subplots(
#     nrows=len(cat_cols),
#     ncols=1,
#     figsize=(10, 5 * len(cat_cols))
# )

# for i, col in enumerate(cat_cols):
#     sns.countplot(
#         data=train_df,
#         x=col,
#         ax=axes[i],
#         color=colors[i]
#     )

# plt.tight_layout()
# plt.savefig("images/cat_data.jpeg", dpi=300)
# plt.show()

# fig, axes = plt.subplots(
#     nrows=len(num_cols),
#     ncols=2,
#     figsize=(20, 5 * len(num_cols))
# )

# for i, col in enumerate(num_cols):
#     sns.histplot(
#         data=train_df,
#         x=col,
#         kde=True,
#         bins=30,
#         ax=axes[i, 0],
#         color=colors[i]
#     )

#     sns.boxplot(
#         data=train_df,
#         x=col,
#         ax=axes[i, 1],
#         color=colors[i]
#     )

# plt.tight_layout()
# plt.savefig("images/num_data.jpeg", dpi=300)
# plt.show()

# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=2,
#     figsize=(10, 5)
# )

# sns.histplot(
#     data=train_df,
#     x=target,
#     kde=True,
#     ax=axes[0]
# )

# sns.boxplot(
#     data=train_df,
#     x=target,
#     ax=axes[1]
# )

# plt.tight_layout()
# plt.savefig("images/target.jpeg", dpi=300)
# plt.show()

# corr_matrix = train_df[num_cols].corr()

# plt.figure(figsize=(10, 5))

# sns.heatmap(corr_matrix, annot=True)

# plt.tight_layout()
# plt.savefig("images/corr_matrix.jpeg", dpi=300)
# plt.show()

for col in num_cols:
    upper = train_df[col].quantile(0.95)
    train_df[col] = np.where(train_df[col] > upper, upper, train_df[col])
    test_df[col] = np.where(test_df[col] > upper, upper, test_df[col])


def feature_engineering(df) -> pd.DataFrame:
    df["Brand_and_Material"] = df["Brand"] + "_" + df["Material"]
    df["Weight_Category"] = pd.qcut(
        df["Weight_Capacity_(kg)"],
        q=4,  # 4 quantiles
        labels=["Light", "Medium", "Heavy", "Very Heavy"]
    )
    return df


train_df = feature_engineering(train_df)
test = feature_engineering(test_df)

for col in num_cols:
    print("=" * 30)
    print(f"Skewness of {col}: {train_df[col].skew():.4f}")

train, eval = train_test_split(
    train_df,
    train_size=0.8,
    random_state=random_state
)

cat_cols = [
    "Brand",
    "Material",
    "Size",
    "Laptop_Compartment",
    "Waterproof",
    "Style",
    "Color",
    "Brand_and_Material",
    "Weight_Category"
]

num_cols = [
    "Compartments",
    "Weight_Capacity_(kg)"
]

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="constant", fill_value="NA")

train[num_cols] = num_imputer.fit_transform(train[num_cols])
eval[num_cols] = num_imputer.transform(eval[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])

train[cat_cols] = cat_imputer.fit_transform(train[cat_cols])
eval[cat_cols] = cat_imputer.transform(eval[cat_cols])
test[cat_cols] = cat_imputer.transform(test[cat_cols])

for df in [train, eval, test]:
    df["Laptop_Compartment"] = df["Laptop_Compartment"].map(
        {
            "No": 0,
            "Yes": 1,
            "NA": 2
        }
    )

    df["Waterproof"] = df["Waterproof"].map(
        {
            "No": 0,
            "Yes": 1,
            "NA": 2
        }
    )

dummy_cols = [
    "Brand",
    "Material",
    "Size",
    "Style",
    "Color",
    "Brand_and_Material",
    "Weight_Category"
]

train = pd.get_dummies(
    data=train,
    prefix_sep="_",
    columns=dummy_cols,
    dtype=np.int64
)

eval = pd.get_dummies(
    data=eval,
    prefix_sep="_",
    columns=dummy_cols,
    dtype=np.int64
)

test = pd.get_dummies(
    data=test,
    prefix_sep="_",
    columns=dummy_cols,
    dtype=np.int64
)

train.columns = train.columns.str.replace(" ", "_")
eval.columns = eval.columns.str.replace(" ", "_")
test.columns = test.columns.str.replace(" ", "_")

ss = StandardScaler()

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

X_train[num_cols] = ss.fit_transform(X_train[num_cols], y=y_train)
X_eval[num_cols] = ss.transform(X_eval[num_cols])

test[num_cols] = ss.transform(test[num_cols])


def objective_XGB(trial) -> float:
    params = {
        'objective': 'reg:squarederror',
        "n_estimators": trial.suggest_int('n_estimators', 500, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        'verbosity': 0,
        'device': 'cuda',
        'n_jobs': -1,
        "eval_metric": "rmse"
    }
    model = XGBRegressor(
        **params,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    rmse = root_mean_squared_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("RMSE: %.4f" % (rmse))
    print("=" * 13)

    return rmse


study_XGB = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study_XGB.optimize(objective_XGB, n_trials=opt_iter)

model_XGB = XGBRegressor(
    **study_XGB.best_params,
    random_state=random_state,
)
model_XGB.fit(X_train, y_train)

y_pred = model_XGB.predict(X_eval)

score_XGB = root_mean_squared_error(y_true=y_eval, y_pred=y_pred)

print(f"XGB score: {score_XGB}")


def objective_LGBM(trial) -> float:
    params = {
        'objective': 'regression',
        "n_estimators": trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbosity': -1,
        'random_state': random_state,
        "metric": "rmse"
    }
    model = LGBMRegressor(
        **params,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    rmse = root_mean_squared_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("RMSE: %.4f" % (rmse))
    print("=" * 13)

    return rmse


study_LGBM = optuna.create_study(direction='minimize')
study_LGBM.optimize(objective_LGBM, n_trials=opt_iter)

model_LGBM = LGBMRegressor(
    **study_LGBM.best_params,
    random_state=random_state,
)
model_LGBM.fit(X_train, y_train)

y_pred = model_LGBM.predict(X_eval)

score_LGBM = root_mean_squared_error(y_true=y_eval, y_pred=y_pred)

print(f"LGBM score: {score_LGBM}")

final_pred = 0.5 * model_XGB.predict(X_eval) + 0.5 * model_LGBM.predict(X_eval)
ensemble_rmse = root_mean_squared_error(y_eval, final_pred)

print(f"Ensemble score: {ensemble_rmse}")

kf = KFold(n_splits=8, shuffle=True, random_state=random_state)

rmse_scores_xgb = []
rmse_scores_lgbm = []

feature_importance_XGB = np.zeros(X_train.shape[1])
feature_importance_LGBM = np.zeros(X_train.shape[1])

for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nFold {fold_num + 1}/{kf.n_splits}")
    print("=" * 50)

    # Split data using indices from KFold
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    print("Training XGBoost...")
    model_XGB.fit(X_fold_train, y_fold_train)
    print("Training LightGBM...")
    model_LGBM.fit(X_fold_train, y_fold_train)

    xgb_pred = model_XGB.predict(X_fold_val)
    lgbm_pred = model_LGBM.predict(X_fold_val)

    xgb_rmse = root_mean_squared_error(y_fold_val, xgb_pred)
    lgbm_rmse = root_mean_squared_error(y_fold_val, lgbm_pred)

    rmse_scores_xgb.append(xgb_rmse)
    rmse_scores_lgbm.append(lgbm_rmse)

    print(f"XGB RMSE: {xgb_rmse:.4f}")
    print(f"LGBM RMSE: {lgbm_rmse:.4f}")

    feature_importance_XGB += model_XGB.feature_importances_
    feature_importance_LGBM += model_XGB.feature_importances_

print("\nFinal Cross-Validation Results:")
print(f"XGBoost Average RMSE: {np.mean(rmse_scores_xgb):.4f} (±{np.std(rmse_scores_xgb):.4f})")
print(f"LightGBM Average RMSE: {np.mean(rmse_scores_lgbm):.4f} (±{np.std(rmse_scores_lgbm):.4f})")

feature_importance_XGB /= kf.n_splits
feature_importance_LGBM /= kf.n_splits

# Create DataFrames
df_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance_XGB
}).sort_values('Importance', ascending=False)

df_importance_lgbm = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance_LGBM
}).sort_values('Importance', ascending=False)

# Plot XGBoost importance
plt.figure(figsize=(12, 6))
plt.barh(df_importance_xgb['Feature'][:20], df_importance_xgb['Importance'][:20], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top 20 Feature Importances - XGBoost')
plt.gca().invert_yaxis()
plt.savefig("images/xgb_importances.jpeg", dpi=300)
plt.show()

# Plot LightGBM importance
plt.figure(figsize=(12, 6))
plt.barh(df_importance_lgbm['Feature'][:20], df_importance_lgbm['Importance'][:20], color='salmon')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Top 20 Feature Importances - LightGBM')
plt.gca().invert_yaxis()
plt.savefig("images/lgbm_importances.jpeg", dpi=300)
plt.show()

y_pred_xgb_test = model_XGB.predict(test)
y_pred_lgbm_test = model_LGBM.predict(test)
ensemble_pred_test = 0.6 * y_pred_xgb_test + 0.4 * y_pred_lgbm_test

submission = pd.DataFrame({
    "id": idx,
    target: ensemble_pred_test
})

submission.to_csv(
    f"submission_Ensemble_{
        datetime.now().strftime("%Y%m%d_%H%M%S")
    }.csv",
    index=False
)

