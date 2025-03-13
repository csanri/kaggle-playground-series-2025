import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
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

train_df_modified["temp_range"] = train_df_modified["maxtemp"] - train_df_modified["mintemp"]
test_df["temp_range"] = test_df["maxtemp"] - test_df["mintemp"]

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

# stats = ["mean", "median"]

# cols = ["cloud", "humidity", "temparature"]

# for col in cols:
#     train_rain_stats = train.groupby("day")[col].agg(stats).reset_index()
#     train_rain_stats.columns = ["day"] + [f"daily_{col}_{stat}" for stat in stats]

#     train = train.merge(train_rain_stats, on="day", how="left")
#     eval = eval.merge(train_rain_stats, on="day", how="left")

# print(train, eval)


def create_group_features(df, train_df=None, combo_list=None, stats=None):
    if train_df is None:
        train_df = df

    new_features = []

    for col in combo_list:
        if isinstance(col, str):
            group_cols = [col]

        group_stats = train_df.groupby(list(group_cols)).agg({
            "day": stats,
            "humidity": stats,
            "cloud": stats,
            "temparature": stats,
            "sunshine": stats,
            "winddirection": stats,
            "windspeed": stats,
            "pressure": stats
        })

        group_stats.columns = [
            f"{"_".join(group_cols)}_{col[0]}_{col[1]}" for col in group_stats.columns
        ]
        group_stats.reset_index()

        df = df.merge(group_stats, on=list(group_cols), how="left")
        new_features += list(group_stats.columns[len(group_cols):])

    return df, new_features


stats = ["mean", "median", "std", "min", "max"]

combos = ["day", "humidity", "cloud", "sunshine"]

train, new_features = create_group_features(train, combo_list=combos, stats=stats)
eval, _ = create_group_features(eval, train_df=train,  combo_list=combos, stats=stats)
test, _ = create_group_features(test_df, train_df=train, combo_list=combos, stats=stats)

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

model_XGB_base = XGBClassifier(
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.5,
    early_stopping_rounds=100,
    eval_metric='error',
    device='cuda',
    random_state=random_state
)

eval_set_base = [(X_eval_base, y_eval_base)]

model_XGB_base.fit(X_train_base, y_train_base, eval_set=eval_set_base)

model_XGB = XGBClassifier(
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.5,
    early_stopping_rounds=100,
    eval_metric='error',
    device='cuda',
    random_state=random_state
)

eval_set = [(X_eval, y_eval)]

model_XGB.fit(X_train, y_train, eval_set=eval_set)

pred_base = model_XGB_base.predict(X_eval_base)
pred = model_XGB.predict(X_eval)

score_XGB_base = accuracy_score(y_true=y_eval_base, y_pred=pred_base)
score_XGB = accuracy_score(y_true=y_eval, y_pred=pred)

print(f"XGB baseline score: {score_XGB_base:.4f} \nXGB modified score: {score_XGB:.4f}")

feature_importance = model_XGB.get_booster().get_score(importance_type='weight')

importance_df = pd.DataFrame({
    'feature': list(feature_importance.keys()),
    'importance': list(feature_importance.values())
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 16))
sns.barplot(x='importance', y='feature', data=importance_df.head(50))
plt.title(f'Top {len(X_train.columns)} Feature Importance')
plt.show()

group_feat_importance = importance_df[importance_df['feature'].str.contains('_mean|_std|_max| _min| _median')]
print("Top performing group features:")
print(group_feat_importance.head(20))

pred_test = model_XGB.predict(test)

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

