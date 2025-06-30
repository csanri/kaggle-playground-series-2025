import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

from scipy.stats import chi2_contingency

from catboost import CatBoostClassifier

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

colors = sns.color_palette("tab10")

random_state = 42
opt_iter = 50

train_df = pd.read_csv("playground-series-s5e6/train.csv")
train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv("playground-series-s5e6/test.csv")

target = "Fertilizer Name"

#############
# DATA INFO #
#############

print("=" * 30)
print(train_df.info())
# NOTE: The target is categotical,
# 2 categorical, and 6 numerical features

print("=" * 30)
print(train_df.describe().T)
# NOTE: Phosphorous has relatively high standard deviation
# compared to its mean

for col in train_df.columns:
    print("=" * 30)
    print(f"Values for {col}")
    print("-" * 30)
    print(train_df[col].value_counts())
# NOTE: 'Soil Type' has 5 different values
# 'Crop Type' has 11 different values
# 'Fertilizer Name' has 7 different values
# Features seem to be evenly distributed

print("=" * 30)
print(train_df.isna().sum())
# NOTE: No missing values

######################
# DATA VISUALIZATION #
######################

cat_cols = train_df.select_dtypes(include="object").columns.to_list()
num_cols = train_df.select_dtypes(exclude="object").columns.to_list()

cat_cols.remove(target)

# Distirbution of categorical features
# fig, axes = plt.subplots(
#     figsize=(20, 12),
#     nrows=1,
#     ncols=2
# )

# for i, col in enumerate(cat_cols):
#     values = train_df[col].value_counts()
#     axes[i].pie(
#         x=values.values,
#         labels=values.index,
#         autopct="%1.1f%%",
#         colors=colors
#     )
#     axes[i].set_title(f'Distribution of {col}', fontsize=14)

# plt.tight_layout()
# plt.savefig("figures/categorical_distribution.png", dpi=300)
# plt.show()
# NOTE: The categorical data is evenly distributed

# Numerical distribution
# fig, axes = plt.subplots(
#     figsize=(20, 12),
#     nrows=2,
#     ncols=3
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
# plt.savefig("figures/numerical_distribution.png", dpi=300)
# plt.show()
# NOTE: The numerical data is evenly distributed

# Target distribution
# plt.figure(figsize=(20, 20))
# values = train_df[target].value_counts()

# plt.pie(
#     x=values.values,
#     labels=values.index,
#     autopct="%1.1f%%",
#     colors=colors
# )
# plt.title("Target Distribution")

# plt.tight_layout()
# plt.savefig("figures/target_distribution.png", dpi=300)
# plt.show()
# NOTE: The numerical data is evenly distributed

# Pair Plot of Numerical data depending on the target
# plt.figure(figsize=(20, 20))

# sns.pairplot(
#     data=train_df,
# )

# plt.tight_layout()
# plt.savefig("figures/pair_plot.png", dpi=300)
# plt.show()

# Correlation Matrix
# plt.figure(figsize=(20, 20))

# corr_matrix = train_df[num_cols].corr()

# sns.heatmap(
#     data=corr_matrix,
#     cmap="coolwarm",
#     fmt=".2f",
#     annot=True
# )

# plt.tight_layout()
# plt.savefig("figures/corr_matrix.png", dpi=300)
# plt.show()
# NOTE: No correlation


# Implementing Cramér's v to see
# correlation between categorical data
# def cramers_v(x, y):
#     confusion_matrix = pd.crosstab(x, y)
#     chi2 = chi2_contingency(confusion_matrix)[0]
#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     return np.sqrt(phi2 / min((k-1), (r-1)))


# cat_cols.append(target)

# corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

# for col1 in cat_cols:
#     for col2 in cat_cols:
#         corr_matrix.loc[col1, col2] = cramers_v(
#             train_df[col1],
#             train_df[col2]
#         )

# plt.figure(figsize=(20, 20))

# sns.heatmap(
#     data=corr_matrix.astype(float),
#     cmap="coolwarm",
#     fmt=".2f",
#     annot=True
# )

# plt.tight_layout()
# plt.savefig("figures/cramers_v_corr.png", dpi=300)
# plt.show()
# NOTE: No correlation

#######################
# FEATURE ENGINEERING #
#######################


def feature_engineering(df) -> pd.DataFrame:
    # Making relation between the environmetal
    # variables and the gas data
    for env in ["Temparature", "Humidity", "Moisture"]:
        for gas in ["Nitrogen", "Potassium", "Phosphorous"]:
            df[f"{gas}_{env}_ratio"] = df[gas] / df[env]

    df["Crop_Moisture"] = df["Crop Type"].astype("str") \
        + "_" + df["Moisture"].astype("str")

    df["Crop_Soil"] = df["Crop Type"].astype("str") \
        + "_" + df["Soil Type"].astype("str")

    df["Crop_Humidity"] = df["Crop Type"].astype("str") \
        + "_" + df["Humidity"].astype("str")

    df["Moisture_Crop_Soil"] = df["Crop Type"].astype("str") \
        + "_" + df["Moisture"].astype("str") \
        + "_" + df["Soil Type"]

    return df


train_df = feature_engineering(test_df)
test_df = feature_engineering(test_df)

cat_cols = train_df.select_dtypes(include="object")

# Label Encoding categorical features
le = LabelEncoder()

for col in cat_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

######

train, eval = train_test_split(
    train_df,
    train_size=0.8,
    random_state=random_state
)

# Scaling numerical features
scaler = MinMaxScaler()

train[num_cols] = scaler.fit_transform(train[num_cols])
eval[num_cols] = scaler.transform(eval[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

# Training a baseline model
model_CAT = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
)

model_CAT.fit(X_train, y_train)
y_pred = model_CAT.predict(X_eval)

acc_score_CAT = accuracy_score(y_true=y_eval, y_pred=y_pred)

print(f"CatBoost accuraccy: {acc_score_CAT}")

# Feature Importance
feature_importance = pd.DataFrame(
    {
        "Feature": X_train.columns.to_list(),
        "Feature Importance": model_CAT.feature_importances_
    }
).sort_values(by="Feature Importance", ascending=False)


def objective_CAT(trial) -> float:
    params = {
        'objective': "MultiClass",  # For classification
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-3, 0.1, log=True
        ),
        'depth': trial.suggest_int('depth', 2, 10),
        'colsample_bylevel': trial.suggest_float(
            'colsample_bylevel', 0.05, 1.0
        ),
        'l2_leaf_reg': trial.suggest_float(
            'l2_leaf_reg', 1e-3, 10.0, log=True
        ),
        'random_strength': trial.suggest_float(
            'random_strength', 1e-5, 10.0, log=True
        ),
        'early_stopping_rounds': trial.suggest_int(
            'early_stopping_rounds', 10, 100
        ),
        'verbose': False,
        'random_state': 42,
        'auto_class_weights': 'Balanced',
        'loss_function': 'MultiClass',
    }

    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_eval, y_eval),
        use_best_model=True,
    )

    y_pred_proba = model.predict_proba(X_eval)

    metric = log_loss(y_eval, y_pred_proba)

    print(f"Trial {trial.number}: Log Loss = {metric:.5f}")

    return metric


study_CAT = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study_CAT.optimize(objective_CAT, n_trials=opt_iter)

model_CAT = CatBoostClassifier(
    **study_CAT.best_params,
)
model_CAT.fit(X_train, y_train)

y_pred = model_CAT.predict_proba(X_eval)

