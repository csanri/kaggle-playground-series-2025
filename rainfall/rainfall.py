import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

# Reading data
train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

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

fig, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(20, 16)
)

axes = axes.flat

for i, (ax, col) in enumerate(zip(axes, num_cols)):
    sns.histplot(
        data=train_df,
        x=col,
        ax=ax,
        kde=True,
        color=colors[i % len(colors)]
    )

plt.tight_layout()
plt.savefig("images/num_cols.jpeg", dpi=300)
plt.show()

corr_matrix = train_df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 12))
sns.heatmap(
    data=corr_matrix,
    annot=True,
    mask=mask,
    fmt=".2f",
    cmap="coolwarm"
)
plt.tight_layout()
plt.savefig("images/corr_matrix.jpeg", dpi=300)
plt.show()

plt.figure(figsize=(20, 12))
sns.countplot(
    data=train_df,
    x=target,
    color=colors[2],
)
plt.tight_layout()
plt.savefig("images/target.jpeg", dpi=300)
plt.show()

##################
# PREPARING DATA #
##################

train_df_modified = train_df.copy()

train_df_modified["temp_range"] = train_df_modified["maxtemp"] - train_df_modified["mintemp"]

stats = ["mean", "nunique", "skew"]

for stat in stats:
    train_df_modified[f"daily_rain_{stat}"] = train_df_modified.groupby("day")["rainfall"].transform(stat)

train_base, eval_base = train_test_split(
    train_df,
    train_size=0.8,
    random_state=random_state
)

train, eval = train_test_split(
    train_df_modified,
    train_size=0.8,
    random_state=random_state
)

X_train_base, y_train_base = train_base.drop(target, axis=1), train_base[target]
X_eval_base, y_eval_base = eval_base.drop(target, axis=1), eval_base[target]

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

ss = StandardScaler()

X_train[num_cols] = ss.fit_transform(X_train[num_cols], y_train)
X_eval[num_cols] = ss.transform(X_eval[num_cols])

########################
# TRAINING AND TESTING #
########################

model_XGB_base = XGBClassifier(random_state=random_state)
model_XGB_base.fit(X_train_base, y_train_base)

model_XGB = XGBClassifier(random_state=random_state)
model_XGB.fit(X_train, y_train)

pred_base = model_XGB_base.predict(X_eval_base)
pred = model_XGB.predict(X_eval)

score_XGB_base = accuracy_score(y_true=y_eval_base, y_pred=pred_base)
score_XGB = accuracy_score(y_true=y_eval, y_pred=pred)

print(f"XGB baseline score: {score_XGB_base:.4f} \nXGB modified score: {score_XGB:.4f}")

