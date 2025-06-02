import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from datetime import datetime

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

colors = sns.color_palette("tab10")

random_state = 42
opt_iter = 250

train_df = pd.read_csv("playground-series-s5e5/train.csv")
train_df.drop("id", axis=1, inplace=True)

test_df = pd.read_csv("playground-series-s5e5/test.csv")
idx = test_df["id"]
test_df = test_df.drop("id", axis=1)

target = "Calories"

#############
# DATA INFO #
#############

print("=" * 30)
print(train_df.info())

print("=" * 30)
print(test_df.info())
# NOTE: Only 1 non numeric column

print("=" * 30)
print(train_df.describe().T)

print("=" * 30)
print(test_df.describe().T)
# NOTE: Calories burnt has relively high standard deviation

for col in train_df.columns:
    print("-" * 30)
    print(train_df[col].value_counts())

for col in test_df.columns:
    print("-" * 30)
    print(test_df[col].value_counts())

print(train_df.isna().sum())
print(test_df.isna().sum())
# NOTE: No missing values

cat_cols = train_df.select_dtypes(include="object").columns.to_list()
num_cols = train_df.select_dtypes(exclude="object").columns.to_list()
num_cols.remove(target)

######################
# DATA VISUALISATION #
######################

# plt.figure(figsize=(10, 18))

# for col in cat_cols:
#     sns.countplot(
#         data=train_df,
#         x=col
#     )

# plt.savefig("figures/cat_cols.png", dpi=300)
# plt.show()
# NOTE: The ratio of males and females is even

# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=3,
#     figsize=(20, 16)
# )

# axes = axes.flat

# for i, (ax, col) in enumerate(zip(axes, num_cols)):
#     sns.histplot(
#         data=train_df,
#         x=col,
#         ax=ax,
#         kde=True,
#         color=colors[i % len(colors)]
#     )

# plt.tight_layout()
# plt.savefig("figures/num_cols.png", dpi=300)
# plt.show()
# NOTE: Body_Temp data looks skewed

# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=3,
#     figsize=(20, 16)
# )

# axes = axes.flat

# for i, (ax, col) in enumerate(zip(axes, num_cols)):
#     sns.scatterplot(
#         data=train_df,
#         x=col,
#         y=target,
#         ax=ax,
#         color=colors[i % len(colors)]
#     )

# plt.tight_layout()
# plt.savefig("figures/scatter_plot.png", dpi=300)
# plt.show()
# NOTE: There is a non linear relationship between Calories and Body_Temp

# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=3,
#     figsize=(20, 16)
# )

# axes = axes.flat

# for i, (ax, col) in enumerate(zip(axes, num_cols)):
#     sns.boxplot(
#         data=train_df,
#         x=col,
#         ax=ax,
#         color=colors[i % len(colors)]
#     )

# plt.tight_layout()
# plt.savefig("figures/boxplot.png", dpi=300)
# plt.show()
# NOTE: Height, Weight, Heart_Rate, Body_Temp has ourliesrs

# plt.figure(figsize=(10, 10))

# sns.histplot(
#     data=train_df,
#     x=target,
#     kde=True
# )

# plt.tight_layout()
# plt.savefig("figures/target_plot.png", dpi=300)
# plt.show()
# NOTE: Calories is skewed

# corr_matrix = train_df[num_cols].corr()

# plt.figure(figsize=(20, 20))

# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="magma"
# )

# plt.tight_layout()
# plt.savefig("figures/corr_matrix.png", dpi=300)
# plt.show()
# NOTE: Duration, Heart_Rate, Body_Temp are highly correlated,
# also Height and Weight

# plt.figure(figsize=(20, 20))

# sns.pairplot(
#     data=train_df,
# )

# plt.tight_layout()
# plt.savefig("figures/pair_plot.png", dpi=300)
# plt.show()
# NOTE: Some features are exponentially correlated

#######################
# FEATURE ENGINEERING #
#######################


def feature_engineering(df) -> pd.DataFrame:
    df["Body_Temp"] = np.log1p(df["Body_Temp"])
    df["BMI"] = df["Weight"] / df["Height"]
    df["HR_to_BMI"] = df["Heart_Rate"] * df["BMI"]
    df["HR_over_time"] = df["Heart_Rate"] / df["Duration"]
    df["BT_over_time"] = df["Body_Temp"] / df["Duration"]
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["Age_Binned"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 70, 80],
        labels=["20_30", "30_40", "40_50", "50_60", "60_70", "70_80"],
        right=False
    )

    # calories_by_age = df.groupby(
    #     ["Age_Binned"]
    # )["Duration"].mean().reset_index()

    # df = pd.merge(
    #     df, calories_by_age,
    #     on="Age_Binned",
    #     how="left",
    #     suffixes=("", "_Age_avg")
    # )

    df["Age_Binned"] = df["Age_Binned"].map({
        "20_30": 0,
        "30_40": 1,
        "40_50": 2,
        "50_60": 3,
        "60_70": 4,
        "70_80": 5
    })

    return df


train_df = feature_engineering(train_df)
test = feature_engineering(test_df)

num_cols = train_df.select_dtypes(exclude="object").columns.to_list()
corr_matrix = train_df[num_cols].corr()

# plt.figure(figsize=(20, 20))

# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="magma"
# )

# plt.tight_layout()
# plt.savefig("figures/corr_matrix_plus_features.png", dpi=300)
# plt.show()
# NOTE: Drop highly correlated feature

# train_df = train_df.drop(
#     [
#         "Height",
#         "Weight"
#     ],
#     axis=1
# )

train, eval = train_test_split(train_df, train_size=0.8)

train[target] = np.log1p(train[target])

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

###############
# BASIC MODEL #
###############

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_eval)
y_pred = np.expm1(y_pred)

print(root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred))

######################
# FEATURE IMPORTANCE #
######################

model_XGB = XGBRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=1000,
    eval_metric="rmsle",
    random_state=random_state
)

model_XGB.fit(X_train, y_train)

importance_score = model_XGB.feature_importances_
features = train_df.columns.to_list()
features.remove(target)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance_score
}).sort_values(by="Importance", ascending=False)

# Selecting the top 8 important features
top_features = importance_df.head(8)["Feature"].values

X_train, y_train = train[top_features], train[target]
X_eval, y_eval = eval[top_features], eval[target]

test = test[top_features]

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

test = scaler.transform(test)

# plt.figure(figsize=(20, 16))

# sns.barplot(
#     data=importance_df,
#     x="Importance",
#     y="Feature",
#     palette="tab10"
# )

# plt.tight_layout()
# plt.savefig("figures/feature_importance.png", dpi=300)
# plt.show()

######


def objective_XGB(trial) -> float:
    params = {
        'objective': 'reg:squarederror',
        "n_estimators": trial.suggest_int('n_estimators', 500, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        'verbosity': 0,
        'device': 'cuda',
        'n_jobs': -1,
        "eval_metric": "rmsle"
    }
    model = XGBRegressor(**params, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_pred = np.expm1(y_pred)

    rmsle = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("RMSLE: %.5f" % (rmsle))
    print("=" * 13)

    return rmsle


study_XGB = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study_XGB.optimize(objective_XGB, n_trials=opt_iter)

model_XGB = XGBRegressor(
    **study_XGB.best_params,
    random_state=random_state
)
model_XGB.fit(X_train, y_train)

y_pred = model_XGB.predict(X_eval)

score_XGB = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

print(f"XGB score: {score_XGB}")

######


def objective_CAT(trial) -> float:
    params = {
        'objective': 'CrossEntropy',
        "iterations": trial.suggest_int('n_estimators', 500, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("max_depth", 1, 12),
        "colsample_bylevel": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        'verbose': -1,
        "loss_function": "rmse",
        "early_stopping_rounds": 50
    }
    model = CatBoostRegressor(**params, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_pred = np.expm1(y_pred)

    rmsle = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("RMSLE: %.5f" % (rmsle))
    print("=" * 13)

    return rmsle


study_CAT = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study_CAT.optimize(objective_CAT, n_trials=opt_iter)

model_CAT = CatBoostRegressor(
    **study_CAT.best_params,
    random_state=random_state
)
model_CAT.fit(X_train, y_train)

y_pred = model_CAT.predict(X_eval)

score_CAT = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

print(f"CAT score: {score_CAT}")


model_LGBM = LGBMRegressor(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=10000,
    random_state=42
)


def objective_LGBM(trial) -> float:
    params = {
        'objective': 'regression',
        "n_estimators": trial.suggest_int('n_estimators', 500, 10000),
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
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_pred = np.expm1(y_pred)

    rmsle = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("MAPE: %.5f" % (rmsle))
    print("=" * 13)

    return rmsle


study_LGBM = optuna.create_study(direction='minimize')
study_LGBM.optimize(objective_LGBM, n_trials=opt_iter)

model_LGBM = LGBMRegressor(
    **study_LGBM.best_params,
    random_state=random_state
)
model_LGBM.fit(X_train, y_train)

y_pred = model_LGBM.predict(X_eval)

score_LGBM = root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred)

print(f"LGBM score: {score_LGBM}")

######

y_test_XGB = model_XGB.predict(test)
y_test_XGB = np.expm1(y_test_XGB)

y_test_CAT = model_CAT.predict(test)
y_test_CAT = np.expm1(y_test_CAT)

y_test_LGBM = model_LGBM.predict(test)
y_test_LGBM = np.expm1(y_test_LGBM)

#####################
# RESIDUAL ANALYSIS #
#####################

# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=2,
#     figsize=(20, 20)
# )

# for i, pred in enumerate([y_pred_XGB, y_pred_CAT]):
#     sns.regplot(
#         x=pred,
#         y=y_eval,
#         line_kws={"color": "r"},
#         ax=axes[0, i]
#     )

# for i, pred in enumerate([y_pred_XGB, y_pred_CAT]):
#     y_res = abs(y_eval - pred)

#     sns.regplot(
#         x=pred,
#         y=y_res,
#         line_kws={"color": "r"},
#         ax=axes[1, i]
#     )

# plt.tight_layout()
# plt.savefig("figures/residual_analysis.png", dpi=300)
# plt.show()

y_test = 0.5 * y_test_XGB + 0.5 * y_test_CAT

submission = pd.DataFrame({
    "id": idx,
    target: y_test_CAT
})

print(submission.head())

submission.to_csv(
    f"submissions/submission_{
        datetime.now().strftime("%Y%m%d_%H%M%S")
    }.csv",
    index=False
)

