import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from datetime import datetime

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

# Reading data
train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv("test.csv")
idx = test_df["id"]
test_df = test_df.drop("id", axis=1)

target = "rainfall"

random_state = 42

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

##################
# PREPARING DATA #
##################

train_df_modified = train_df.copy()

train_base, eval_base = train_test_split(
    train_df,
    train_size=0.8,
    random_state=random_state
)

# for col in num_cols:
#     train_df_modified[f"{col}_bin"] = pd.qcut(train_df_modified[col], q=5, labels=False)

train, eval = train_test_split(
    train_df_modified,
    train_size=0.8,
    random_state=random_state
)


def generate_months(df) -> pd.DataFrame:
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

    return df


# train_df_modified["temp_range"] = train_df_modified["maxtemp"] - train_df_modified["mintemp"]
# test_df["temp_range"] = test_df["maxtemp"] - test_df["mintemp"]

train = generate_months(train)
eval = generate_months(eval)
test = generate_months(test_df)

stats = ["mean", "median"]

group = "month"
cols = ["cloud", "humidity", "temparature"]

for col in cols:
    train_rain_stats = train.groupby(group)[col].agg(stats).reset_index()
    train_rain_stats.columns = [group] + [f"{group}_{col}_{stat}" for stat in stats]

    train = train.merge(train_rain_stats, on=group, how="left")
    eval = eval.merge(train_rain_stats, on=group, how="left")
    test = test.merge(train_rain_stats, on=group, how="left")


def onehot(df) -> pd.DataFrame:
    df = pd.get_dummies(
        df,
        prefix_sep="_",
        columns=["month"],
        dtype=np.int32,
    )
    return df


train = onehot(train)
eval = onehot(eval)
test = onehot(test)


def create_group_features(df, train_df=None, combo_list=None, stats=None):
    if train_df is None:
        train_df = df

    new_features = []

    for col in combo_list:
        if isinstance(col, str):
            group_cols = [col]

        group_stats = train_df.groupby(list(group_cols)).agg({
            "humidity": stats,
            "cloud": stats,
            "pressure": stats
        })

        group_stats.columns = [
            f"{"_".join(group_cols)}_{col[0]}_{col[1]}" for col in group_stats.columns
        ]
        group_stats.reset_index()

        df = df.merge(group_stats, on=list(group_cols), how="left")
        new_features += list(group_stats.columns[len(group_cols):])

    return df, new_features


stats = ["mean", "median", "std"]

combos = ["day"]

# train, new_features = create_group_features(train, combo_list=combos, stats=stats)
# eval, _ = create_group_features(eval, train_df=train,  combo_list=combos, stats=stats)
# test, _ = create_group_features(test_df, train_df=train, combo_list=combos, stats=stats)

X_train_base, y_train_base = train_base.drop(target, axis=1), train_base[target]
X_eval_base, y_eval_base = eval_base.drop(target, axis=1), eval_base[target]

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

ss = StandardScaler()

X_train[num_cols] = ss.fit_transform(X_train[num_cols], y_train)
X_eval[num_cols] = ss.transform(X_eval[num_cols])
test[num_cols] = ss.transform(test[num_cols])

########################
# TRAINING AND TESTING #
########################

# model_XGB_base = XGBClassifier(
#     n_estimators=10000,
#     learning_rate=0.02,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.5,
#     early_stopping_rounds=100,
#     eval_metric='auc',
#     device='cuda',
#     random_state=random_state
# )

logreg_model_base = LogisticRegression(
    max_iter=1000,
    random_state=42,
    penalty='l2',
    class_weight='balanced',
    solver='liblinear'
)

# eval_set_base = [(X_eval_base, y_eval_base)]

# model_XGB_base.fit(X_train_base, y_train_base, eval_set=eval_set_base)
logreg_model_base.fit(X_train_base, y_train_base)

# model_XGB = XGBClassifier(
#     n_estimators=10000,
#     learning_rate=0.02,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.5,
#     early_stopping_rounds=100,
#     eval_metric='auc',
#     device='cuda',
#     random_state=random_state
# )

logreg_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    penalty='l2',
    class_weight='balanced',
    solver='liblinear'
)

# eval_set = [(X_eval, y_eval)]

# model_XGB.fit(X_train, y_train, eval_set=eval_set)
logreg_model.fit(X_train, y_train)

# pred_base = model_XGB_base.predict(X_eval_base)
# pred = model_XGB.predict(X_eval)

pred_base = logreg_model_base.predict_proba(X_eval_base)[:, 1]
pred = logreg_model.predict_proba(X_eval)[:, 1]

pred_base = (pred_base >= 0.6).astype("int")
pred = (pred >= 0.6).astype("int")

roc_auc_XGB_base = roc_auc_score(y_true=y_eval_base, y_score=pred_base)
roc_auc_XGB = roc_auc_score(y_true=y_eval, y_score=pred)

# feature_importance = model_XGB.get_booster().get_score(importance_type="weight")

# importance_df = pd.DataFrame({
#     'feature': list(feature_importance.keys()),
#     'importance': list(feature_importance.values())
# }).sort_values('importance', ascending=False)

# plt.figure(figsize=(12, 16))
# sns.barplot(x='importance', y='feature', data=importance_df.head(50))
# plt.title(f'Top {len(X_train.columns)} Feature Importance')
# plt.show()

# print("=" * 30)
# print(f"XGB baseline AUC-ROC: {roc_auc_XGB_base:.4f} \nXGB modified AUC-ROC: {roc_auc_XGB:.4f}")
# print("=" * 30)


print(f"LogReg baseline AUC-ROC: {roc_auc_XGB_base:.4f} \nLogReg modified AUC-ROC: {roc_auc_XGB:.4f}")

# pred_test = model_XGB.predict(test)
pred_test = logreg_model.predict_proba(test)[:, 1]
pred_test = (pred_test >= 0.6).astype("int")

submission = pd.DataFrame({
    "id": idx,
    target: pred_test
})

submission.to_csv(
    f"submissions/submission_{
        datetime.now().strftime("%Y%m%d_%H%M%S")
    }.csv",
    index=False
)

