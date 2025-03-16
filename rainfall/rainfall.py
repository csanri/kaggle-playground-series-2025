import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from datetime import datetime

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

# Reading data
train_df = pd.read_csv("train.csv")
# train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv("test.csv")
idx = test_df["id"]
test_df = test_df.drop("id", axis=1)

target = "rainfall"
random_state = 42
treshold = 0.5

#################
# DATA ANALYSIS #
#################
print(train_df.info())
print(50 * "=")
print(train_df.describe().transpose())
print(50 * "=")
print(train_df.isna().sum())
print(test_df.isna().sum())
print(50 * "=")

test_df["winddirection"] = test_df["winddirection"].fillna(test_df["winddirection"].median())

print(50 * "=")
print(test_df.isna().sum())
print(50 * "=")

for col in train_df.columns:
    print(train_df[col].value_counts())
    print(50 * "-")

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
num_cols = train_df.select_dtypes(exclude="object").columns.tolist()

num_cols.remove(target)

colors = sns.color_palette("tab10")

# fig, axes = plt.subplots(
#     nrows=3,
#     ncols=4,
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
# plt.savefig("images/num_cols.jpeg", dpi=300)
# plt.show()

# corr_matrix = train_df.corr()

# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# plt.figure(figsize=(12, 12))
# sns.heatmap(
#     data=corr_matrix,
#     annot=True,
#     mask=mask,
#     fmt=".2f",
#     cmap="coolwarm"
# )
# plt.tight_layout()
# plt.savefig("images/corr_matrix.jpeg", dpi=300)
# plt.show()

# plt.figure(figsize=(20, 12))
# sns.countplot(
#     data=train_df,
#     x=target,
#     color=colors[2],
# )
# plt.tight_layout()
# plt.savefig("images/target.jpeg", dpi=300)
# plt.show()

# fig, axes = plt.subplots(
#     nrows=6,
#     ncols=2,
#     figsize=(12, 20)
# )

# axes = axes.flat

# for i, (ax, col) in enumerate(zip(axes, num_cols)):
#     sns.boxplot(
#         data=train_df,
#         x=col,
#         ax=ax,
#         color=colors[i % len(colors)],
#     )

# plt.tight_layout()
# plt.savefig("images/boxplots.jpeg", dpi=300)
# plt.show()

plt.figure(figsize=(20, 16))
plt.plot(train_df["id"], train_df["day"])
plt.show()

##################
# PREPARING DATA #
##################


def add_features(df) -> pd.DataFrame:
    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    cum_days = [sum(days_in_months[:i+1]) for i in range(len(days_in_months))]

    bins = [0] + cum_days
    labels = [i for i in range(1, 13)]

    df["month"] = pd.cut(
        df["day"],
        bins=bins,
        labels=labels,
        right=True
    )

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365)

    df["cloud_lag1"] = df["cloud"].shift(-1).fillna(0)
    df["temp_lag1"] = df["temparature"].shift(-1).fillna(0)
    df["sunshine_lag1"] = df["sunshine"].shift(-1).fillna(0)

    df["cloud_roll_14"] = df["cloud"].rolling(
        window=14,
        min_periods=1
    ).mean().bfill()

    df["temp_roll_14"] = df["temparature"].rolling(
        window=14,
        min_periods=1
    ).mean().bfill()

    df["sunshine_roll_14"] = df["sunshine"].rolling(
        window=14,
        min_periods=1
    ).mean().bfill()

    df["cloud_humidity"] = (df["cloud"] * df["humidity"]).fillna(0)
    df["sunshine_cloud"] = (df["sunshine"] / df["cloud"] + 1e-4).fillna(0)

    df["cloud_day_sin"] = np.sin(2 * df["day"] / 365 * df["cloud"] * np.pi)
    df["cloud_day_cos"] = np.cos(2 * df["day"] / 365 * df["cloud"] * np.pi)

    df["sunshine_day_sin"] = np.sin(2 * df["day"] / 365 * df["sunshine"] * np.pi)
    df["sunshine_day_cos"] = np.cos(2 * df["day"] / 365 * df["sunshine"] * np.pi)

    df["humidity_day_sin"] = np.sin(2 * df["day"] / 365 * df["humidity"] * np.pi)
    df["humidity_day_cos"] = np.cos(2 * df["day"] / 365 * df["humidity"] * np.pi)

    df["temp_range"] = df["maxtemp"] - df["mintemp"]

    return df


train = add_features(train_df)
test = add_features(test_df)

test["sunshine_cloud"].replace(np.inf, 0, inplace=True)

stats = ["mean", "median"]

group = "day"
cols = ["cloud", "humidity", "temparature"]

for col in cols:
    train_rain_stats = train.groupby(group)[col].agg(stats).reset_index()
    train_rain_stats.columns = [group] + [f"{group}_{col}_{stat}" for stat in stats]

    train = train.merge(train_rain_stats, on=group, how="left")
    test = test.merge(train_rain_stats, on=group, how="left")


def onehot(df) -> pd.DataFrame:
    df = pd.get_dummies(
        df,
        prefix_sep="_",
        columns=["month"],
        dtype=np.int32,
    )
    return df


# corr_matrix = train.corr()

# plt.figure(figsize=(12, 12))
# sns.heatmap(
#     data=corr_matrix,
#     cmap="coolwarm"
# )
# plt.tight_layout()
# plt.savefig("images/corr_matrix.jpeg", dpi=300)
# plt.show()

cols_to_drop = [
    "day", "windspeed", "dewpoint",
    "mintemp", "maxtemp", "pressure"
]

train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)

num_cols = train.select_dtypes(exclude="object").columns.tolist()
num_cols.remove("rainfall")

for df in [train, test]:
    df = onehot(df)

X_train, y_train = train.drop(target, axis=1), train[target]

mms = MinMaxScaler()

X_train[num_cols] = mms.fit_transform(X_train[num_cols], y_train)
test[num_cols] = mms.transform(test[num_cols])

########################
# TRAINING AND TESTING #
########################

logreg_model = LogisticRegression(
    max_iter=1000,
    random_state=random_state,
    penalty='l2',
    class_weight='balanced',
    solver='liblinear'
)

tsvc = TimeSeriesSplit(n_splits=5)

roc_auc_scores = []

fold = 1

for train_idx, val_idx in tsvc.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    logreg_model.fit(X_train_fold, y_train_fold)

    pred_val = logreg_model.predict_proba(X_val_fold)[:, 1]
    pred_val = (pred_val >= treshold).astype("int")

    roc_auc = roc_auc_score(y_true=y_val_fold, y_score=pred_val)
    roc_auc_scores.append(roc_auc)

    print(f"Fold {fold} - ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_val_fold, pred_val))
    print("=" * 60)

    fold += 1


print(f"Corss-Val score: {np.mean(roc_auc_scores):.4f}")

# pred_test = logreg_model.predict_proba(test)[:, 1]
# pred_test = (pred_test >= treshold).astype("int")

# submission = pd.DataFrame({
#     "id": idx,
#     target: pred_test
# })

# submission.to_csv(
#     f"submissions/submission_{
#         datetime.now().strftime("%Y%m%d_%H%M%S")
#     }.csv",
#     index=False
# )

