import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "Calories"

colors = sns.color_palette("tab10")

train_df = pd.read_csv("playground-series-s5e5/train.csv")
train_df.drop("id", axis=1, inplace=True)

#############
# DATA INFO #
#############

print("=" * 30)
print(train_df.info())
# NOTE: Only 1 non numeric column

print("=" * 30)
print(train_df.describe().T)
# NOTE: Calories burnt has relively high standard deviation

for col in train_df.columns:
    print("-" * 30)
    print(train_df[col].value_counts())

print(train_df.isna().sum())
# NOTE: No missing values

cat_cols = train_df.select_dtypes(include="object").columns.to_list()
num_cols = train_df.select_dtypes(exclude="object").columns.to_list()
num_cols.remove(target)

plt.figure(figsize=(10, 18))

######################
# DATA VISUALISATION #
######################

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

train_df["Body_Temp"] = np.log1p(train_df["Body_Temp"])
train_df["BMI"] = train_df["Weight"] / train_df["Height"]
train_df["HR_to_BMI"] = train_df["Heart_Rate"] * train_df["BMI"]
train_df["HR_over_time"] = train_df["Heart_Rate"] / train_df["Duration"]
train_df["BT_over_time"] = train_df["Body_Temp"] / train_df["Duration"]
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})

train_df["Age_Binned"] = pd.cut(
    train_df["Age"],
    bins=[20, 30, 40, 50, 60, 70, 80],
    labels=["20_30", "30_40", "40_50", "50_60", "60_70", "70_80"],
    right=False
)

calories_by_age = train_df.groupby(
    ["Age_Binned"]
)["Duration"].mean().reset_index()

train_df = pd.merge(
    train_df, calories_by_age,
    on="Age_Binned",
    how="left",
    suffixes=("", "_Age_avg")
)

train_df["Age_Binned"] = train_df["Age_Binned"].map({
    "20_30": 0,
    "30_40": 1,
    "40_50": 2,
    "50_60": 3,
    "60_70": 4,
    "70_80": 5
})

train, eval = train_test_split(train_df, train_size=0.8)

train[target] = np.log1p(train[target])

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

##################
# MODEL TRAINING #
##################

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_eval)
y_pred = np.expm1(y_pred)

print(root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred))

model_XGB = XGBRegressor(
    max_depth=6,
    learning_rate=0.01,
    n_estimators=1000
)

model_XGB.fit(X_train, y_train)
y_pred = model_XGB.predict(X_eval)
y_pred = np.expm1(y_pred)

importance_score = model_XGB.feature_importances_
features = train_df.columns.to_list()
features.remove(target)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance_score
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(20, 16))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    palette="tab10"
)

plt.tight_layout()
plt.show()

print(importance_score)
print(root_mean_squared_log_error(y_true=y_eval, y_pred=y_pred))

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(20, 20)
)

sns.regplot(
    x=y_pred,
    y=y_eval,
    line_kws={"color": "r"},
    ax=axes[0]
)

y_res = abs(y_eval - y_pred)

sns.regplot(
    x=y_pred,
    y=y_res,
    line_kws={"color": "r"},
    ax=axes[1]
)

plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 20))

sns.histplot(y_res, kde=True)

plt.tight_layout()
plt.show()

