import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

colors = sns.color_palette("tab10")

# Reading data
train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv("test.csv")
idx = test_df["id"]
test_df = test_df.drop("id", axis=1)

target = "rainfall"
random_state = 42

# ============================
# DATA ANALYSIS
# ============================
print(train_df.info())
print(50 * "=")
print(train_df.describe().T)
print(50 * "=")
print(train_df.isna().sum())
print(test_df.isna().sum())
print(50 * "=")

test_df["winddirection"] = test_df["winddirection"].fillna(
    test_df["winddirection"].median()
)

print(50 * "=")
print(test_df.isna().sum())
print(50 * "=")

for col in train_df.columns:
    print(train_df[col].value_counts())
    print(50 * "-")

# There are no categorical columns
num_cols = train_df.select_dtypes(exclude="object").columns.tolist()

num_cols.remove(target)

# Histograms of features
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

# Correlation matrix masked to see only the lower triangle
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

# Distribution of the target
# plt.figure(figsize=(20, 12))
# sns.countplot(
#     data=train_df,
#     x=target,
#     color=colors[2],
# )
# plt.tight_layout()
# plt.savefig("images/target.jpeg", dpi=300)
# plt.show()

# Boxplots of numerical features
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

# Using KDE plots to see wheter a feature is predictive or not
# fig, axes = plt.subplots(
#     nrows=6,
#     ncols=2,
#     figsize=(12, 20)
# )

# axes = axes.flat

# for i, (ax, col) in enumerate(zip(axes, num_cols)):
#     sns.kdeplot(
#         data=train_df,
#         x=col,
#         hue=target,
#         ax=ax,
#         color=colors[i % len(colors)]
#     )

# plt.tight_layout()
# plt.savefig("images/kde_plots.jpeg", dpi=300)
# plt.show()

# The day data has errors
# plt.figure(figsize=(20, 16))
# plt.plot(train_df["id"], train_df["day"])
# plt.savefig("images/day_mismatch.jpeg", dpi=300)
# plt.show()

expected_pattern = np.tile(np.arange(1, 366), 6)
train_df["day"] = expected_pattern[:len(train_df)]

# plt.figure(figsize=(20, 16))
# plt.plot(train_df["id"], train_df["day"])
# plt.savefig("images/day_mismatch_corrected.jpeg", dpi=300)
# plt.show()

# ============================
# PREPARING DATA
# ============================

print(train_df.skew())


def add_features(df) -> pd.DataFrame:
    # Months from days
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

    # Day sin/cos for periodic features
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365)

    # Last day's values for notable features
    df["cloud_lag1"] = df["cloud"].shift(-1).fillna(0)
    df["temp_lag1"] = df["temparature"].shift(-1).fillna(0)
    df["sunshine_lag1"] = df["sunshine"].shift(-1).fillna(0)

    # Rolling mean for notable features
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

    # 'Cloud' feature is skewed
    df["cloud_log"] = np.log1p(df["cloud"])

    return df


train = add_features(train_df)
test = add_features(test_df)

# Test data has an inf value for some reason
test["sunshine_cloud"].replace(np.inf, 0, inplace=True)

stats = ["mean", "median", "min", "max", "std"]

group = "day"
cols = ["cloud", "humidity", "sunshine"]

# Aggregate statistical features on daily basis
for col in cols:
    train_rain_stats = train.groupby(group)[col].agg(stats).reset_index()
    train_rain_stats.columns = [group] + [f"{group}_{col}_{stat}" for stat in stats]

    train = train.merge(train_rain_stats, on=group, how="left")
    test = test.merge(train_rain_stats, on=group, how="left")


def onehot(df, column) -> pd.DataFrame:
    df = pd.get_dummies(
        df,
        prefix_sep="_",
        columns=[column],
        dtype=np.int32,
    )
    return df


# Onehot encoding month features
train = onehot(train, "month")
test = onehot(test, "month")

# plt.figure(figsize=(12, 12))
# sns.heatmap(
#     data=corr_matrix,
#     cmap="coolwarm"
# )
# plt.tight_layout()
# plt.savefig("images/corr_matrix.jpeg", dpi=300)
# plt.show()

num_cols = train.select_dtypes(exclude="object").columns.tolist()
num_cols.remove("rainfall")

X_train, y_train = train.drop(target, axis=1), train[target]

mms = MinMaxScaler()

X_train[num_cols] = mms.fit_transform(X_train[num_cols], y_train)
test[num_cols] = mms.transform(test[num_cols])

# Using RFC to determine which features are important
rf = RandomForestClassifier(
    n_estimators=1000,
    random_state=random_state,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

important_features = np.argsort(rf.feature_importances_)[::-1][:20]

selected_features = X_train.columns[important_features].tolist()

corr_matrix = train[selected_features].corr().abs()

# Dropping columns which have high correlation to each other
# to reduce redundancy
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_correlation = [column for column in upper.columns if any(upper[column] > 0.85)]

final_features = [f for f in selected_features if f not in high_correlation]

print(final_features)

X_train = X_train[final_features]
test = test[final_features]

# ============================
# TRAINING AND TESTING
# ============================

logreg_model = LogisticRegression(
    max_iter=1000,
    random_state=random_state,
    penalty='l2',
    class_weight='balanced',
    solver='liblinear'
)

svc_model = SVC(
    kernel="linear",
    probability=True,
    class_weight="balanced",
    random_state=random_state
)

tsvc = TimeSeriesSplit(n_splits=7)

roc_auc_scores_logreg = []
roc_auc_scores_svc = []
roc_auc_scores_mean = []

fold = 1

# Time-Series split CV
for train_idx, val_idx in tsvc.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    logreg_model.fit(X_train_fold, y_train_fold)
    svc_model.fit(X_train_fold, y_train_fold)

    proba_val_logreg = logreg_model.predict_proba(X_val_fold)[:, 1]
    pred_val_logreg = (proba_val_logreg >= 0.5).astype("int")

    proba_val_svc = svc_model.predict_proba(X_val_fold)[:, 1]
    pred_val_svc = (proba_val_svc >= 0.5).astype("int")

    proba_val_mean = proba_val_logreg * 0.5 + proba_val_svc * 0.5
    pred_val_mean = (proba_val_mean >= 0.5).astype("int")

    roc_auc_logreg = roc_auc_score(y_true=y_val_fold, y_score=proba_val_logreg)
    roc_auc_scores_logreg.append(roc_auc_logreg)

    roc_auc_svc = roc_auc_score(y_true=y_val_fold, y_score=proba_val_svc)
    roc_auc_scores_svc.append(roc_auc_svc)

    roc_auc_mean = roc_auc_score(y_true=y_val_fold, y_score=proba_val_mean)
    roc_auc_scores_mean.append(roc_auc_mean)

    print("%-26s" % (f"Fold {fold}"))
    print("-" * 26)
    print("%-15s | %.4f |" % ("ROC-AUC LogReg:", roc_auc_logreg))
    print("-" * 26)
    print("%-15s | %.4f |" % ("ROC-AUC SVC:", roc_auc_svc))
    print("-" * 26)
    print("%-15s | %.4f |" % ("ROC-AUC Mean:", roc_auc_mean))
    print("=" * 26)

    fold += 1


print(f"Corss-Val score LogReg: {np.mean(roc_auc_scores_logreg):.4f}")
print(f"Corss-Val score SVC: {np.mean(roc_auc_scores_svc):.4f}")
print(f"Corss-Val score Mean: {np.mean(roc_auc_scores_mean):.4f}")

pred_test_logreg = logreg_model.predict_proba(test)[:, 1]
pred_test_svc = svc_model.predict_proba(test)[:, 1]

pred_test = pred_test_logreg * 0.5 + pred_test_svc * 0.5

submission = pd.DataFrame({
    "id": idx,
    target: pred_test
})

print(submission.head())

submission.to_csv(
    f"submissions/submission_{
        datetime.now().strftime("%Y%m%d_%H%M%S")
    }.csv",
    index=False
)

