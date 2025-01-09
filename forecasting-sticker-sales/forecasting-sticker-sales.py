import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "num_sold"
random_state = 42

train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

# Getting some basic info about the data
print("=" * 30)
print(train_df.info())
print("=" * 30)
print(train_df.describe())

for col in train_df:
    print("=" * 30)
    print(col)
    print(train_df[col].value_counts())

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("date")

colors = sns.color_palette("tab10")

fig, axes = plt.subplots(
    nrows=len(cat_cols),
    ncols=1,
    figsize=(10, 10 * len(cat_cols))
)

for i, col in enumerate(cat_cols):
    sns.countplot(
        data=train_df,
        x=col,
        ax=axes[i],
        color=colors[i]
    )

plt.tight_layout()
plt.savefig("images/cat_data.jpeg", dpi=300)
plt.show()

plt.figure(figsize=(20, 16))

sns.histplot(
    data=train_df,
    x=target,
    color=colors[0],
    kde=True
)

plt.savefig("images/num_sold.jpeg", dpi=300)
plt.show()

# Checking for missing data
plt.figure(figsize=(20, 16))

sns.heatmap(train_df.isna(), cbar=False, yticklabels=False)
plt.title("Plot of missing values")

plt.savefig("images/missing_values.jpeg", dpi=300)
plt.show()

# The target data has missing values
num_sold_groups = train_df.groupby(cat_cols)[target].count().reset_index()
num_sold_groups = num_sold_groups.pivot_table(
    index="country",
    columns="product",
    values="num_sold",
    aggfunc="sum"
)

plt.figure(figsize=(20, 16))

sns.heatmap(num_sold_groups, annot=True, fmt=".0f")

plt.savefig("images/num_sold_heatmap.jpeg", dpi=300)
plt.show()

train_df["date"] = pd.to_datetime(
    train_df["date"],
    format="ISO8601"
)

countries = train_df["country"].unique()

ncols = 2
nrows = (len(countries) + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(20, 10 * nrows)
)

axes = axes.flatten()

for i, country in enumerate(countries):
    sns.lineplot(
        train_df.loc[train_df["country"] == country],
        x="date",
        y=target,
        ax=axes[i],
        color=colors[i]
    )
    print(country)

plt.tight_layout()
plt.savefig("images/country_lineplot.jpeg", dpi=300)
plt.show()


def date(df):
    df["date_int"] = train_df["date"].astype(np.int64) / 10**9

    df['Year'] = df['date'].dt.year
    df['Day'] = df['date'].dt.day
    df['Month'] = df['date'].dt.month

    df['Month_name'] = df['date'].dt.month_name()
    df['Day_of_week'] = df['date'].dt.day_name()
    df['Week'] = df['date'].dt.isocalendar().week

    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])

    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

    # df['Group'] = (df['Year']-2020)*48 + df['Month']*4 + df['Day']//7

    return df


# train_df = date(train_df)

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("date")

train_df = pd.get_dummies(
    train_df,
    columns=cat_cols,
    prefix_sep="_",
    drop_first=True,
    dtype="int"
)

train_df["date"] = pd.to_datetime(
    train_df["date"],
    format="ISO8601"
).astype(np.int64) / 10**9

train_df = train_df.dropna()

train_df.columns = train_df.columns.str.replace(" ", "_")

train, eval = train_test_split(train_df, train_size=0.8)

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

model_XGB = XGBRegressor(
    n_estimators=1000,
    random_state=random_state
)

model_XGB.fit(X_train, y_train)

y_pred = model_XGB.predict(X_eval)

score_XGB = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)

print(f"XGB score: {score_XGB}")

model_LGBM = LGBMRegressor(
    n_estimators=1000,
    random_state=random_state
)

model_LGBM.fit(X_train, y_train)

y_pred = model_LGBM.predict(X_eval)

score_LGBM = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)

print(f"LGBM score: {score_LGBM}")

