import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "num_sold"
random_state = 42
opt_iter = 30

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

# plt.figure(figsize=(20, 16))

# sns.histplot(
#     data=train_df,
#     x=target,
#     color=colors[0],
#     kde=True
# )

# plt.savefig("images/num_sold.jpeg", dpi=300)
# plt.show()

# # Checking for missing data
# plt.figure(figsize=(20, 16))

# sns.heatmap(train_df.isna(), cbar=False, yticklabels=False)
# plt.title("Plot of missing values")

# plt.savefig("images/missing_values.jpeg", dpi=300)
# plt.show()

# The target data has missing values
num_sold_groups = train_df.groupby(cat_cols)[target].count().reset_index()
num_sold_groups = num_sold_groups.pivot_table(
    index="country",
    columns="product",
    values="num_sold",
    aggfunc="sum"
)

# plt.figure(figsize=(20, 16))

# sns.heatmap(num_sold_groups, annot=True, fmt=".0f")

# plt.savefig("images/num_sold_heatmap.jpeg", dpi=300)
# plt.show()

train_df["date"] = pd.to_datetime(
    train_df["date"],
    format="ISO8601"
)

countries = train_df["country"].unique()

ncols = 2
nrows = (len(countries) + ncols - 1) // ncols

# fig, axes = plt.subplots(
#     nrows=nrows,
#     ncols=ncols,
#     figsize=(5 * ncols, 5 * nrows)
# )

# axes = axes.flatten()

# for i, country in enumerate(countries):
#     sns.lineplot(
#         train_df.loc[train_df["country"] == country],
#         x="date",
#         y=target,
#         ax=axes[i],
#         color=colors[i],
#         errorbar=None
#     )
#     print(country)

# plt.tight_layout()
# plt.savefig("images/country_lineplot.jpeg", dpi=300)
# plt.show()

products = train_df["product"].unique()

# ncols = 2
# nrows = (len(products) + ncols - 1) // ncols

# fig, axes = plt.subplots(
#     nrows=nrows,
#     ncols=ncols,
#     figsize=(5 * ncols, 5 * nrows)
# )

# axes = axes.flatten()

# for i, product in enumerate(products):
#     sns.lineplot(
#         train_df.loc[train_df["product"] == product],
#         x="date",
#         y=target,
#         ax=axes[i],
#         color=colors[i],
#         errorbar=None
#     )
#     print(product)

# plt.tight_layout()
# plt.savefig("images/product_lineplot.jpeg", dpi=300)
# plt.show()

stores = train_df["store"].unique()

# ncols = 3
# nrows = (len(stores) + ncols - 1) // ncols

# fig, axes = plt.subplots(
#     nrows=nrows,
#     ncols=ncols,
#     figsize=(5 * ncols, 5 * nrows)
# )

# axes = axes.flatten()

# for i, store in enumerate(stores):
#     sns.lineplot(
#         train_df.loc[train_df["store"] == store],
#         x="date",
#         y=target,
#         ax=axes[i],
#         color=colors[i],
#         errorbar=None
#     )
#     print(store)

# plt.tight_layout()
# plt.savefig("images/store_lineplot.jpeg", dpi=300)
# plt.show()

country_weights = train_df.groupby("country")[
    target
].sum()/train_df[
    target
].sum()

country_weights = country_weights.reset_index()

# plt.figure(figsize=(20, 16))

# sns.barplot(
#     country_weights,
#     x="country",
#     y=target
# )

# plt.tight_layout()
# plt.savefig("images/country_wights.jpeg", dpi=300)
# plt.show()

weight_over_time = train_df.groupby(["date", "country"])[
    target
].sum()/train_df.groupby("date")[
    target
].sum()

weight_over_time = weight_over_time.reset_index()

# plt.figure(figsize=(20, 16))

# sns.lineplot(
#     weight_over_time,
#     x="date",
#     y=target,
#     hue="country"
# )

# plt.tight_layout()
# plt.savefig("images/country_wights_over_time.jpeg", dpi=300)
# plt.show()

"""
The weights change over time,
probably correlating with GPD
"""

gdp = pd.read_csv("API_NY.GDP.PCAP.CD_DS2_en_csv_v2_76.csv", skiprows=4)
gdp = gdp[gdp["Country Name"].isin(countries)]

min_date = min(train_df["date"]).year
max_date = max(train_df["date"]).year

years = [str(i) for i in range(min_date, max_date + 1)]

gdp = gdp[["Country Name"] + years]

for year in years:
    gdp[year] = gdp[year]/gdp[year].sum()

gdp = gdp.melt(
    id_vars=["Country Name"],
    var_name="Year",
    value_name="GDP per Capita"
)

weight_over_time["date"] = weight_over_time["date"].dt.year

gdp["Year"] = pd.to_datetime(
    gdp["Year"],
    format="ISO8601"
).dt.year

merged_df = pd.merge(weight_over_time, gdp,
                     left_on=["date", "country"],
                     right_on=["Year", "Country Name"],
                     how="left"
                     )

# plt.figure(figsize=(20, 16))

# sns.lineplot(
#     merged_df,
#     x="date",
#     y=target,
#     hue="country",
# )

# sns.lineplot(
#     merged_df,
#     x="Year",
#     y="GDP per Capita",
#     hue="country",
#     legend=False,
#     palette=["black"]*len(countries)
# )

# plt.tight_layout()
# plt.savefig("images/country_weights_and_gdp_over_time.jpeg", dpi=300)
# plt.show()

year_avg = train_df.copy()
year_avg["year"] = year_avg["date"].dt.year
year_avg = year_avg.groupby(["year", "country"])[target].mean().reset_index()

for country in countries:
    na_values = train_df[(train_df[target].isna()) & (train_df["country"] == country)]
    for index, row in na_values.iterrows():
        year = row["date"].year
        matching = year_avg[(year_avg["country"] == country) & (year_avg["year"] == year)]
        if not matching.empty:
            value = matching["num_sold"].values[0]
            train_df.loc[index, target] = value

print(f"Missing values remaining: {train_df[target].isna().sum()}")
print(train_df[target].value_counts())


def date(df):
    df["Year"] = df["date"].dt.year
    df["Day"] = df["date"].dt.day
    df["Month"] = df["date"].dt.month

    # df["Month_name"] = df["date"].dt.month_name()
    # df["Day_of_week"] = df["date"].dt.day_name()
    # df["Week"] = df["date"].dt.isocalendar().week

    df["Year_sin"] = np.sin(2 * np.pi * df["Year"])
    df["Year_cos"] = np.cos(2 * np.pi * df["Year"])

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)

    df.drop(["Year", "Day", "Month"], axis=1, inplace=True)

    return df


# train_df = date(train_df)

cat_cols = train_df.select_dtypes(include="object").columns.tolist()

train_df["lag_7"] = train_df.groupby(["country", "product", "store"])["num_sold"].shift(7)
train_df["rolling_mean_30"] = train_df.groupby(["country", "product", "store"])["num_sold"].transform(lambda x: x.rolling(30, min_periods=1).mean())

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

train_df.columns = train_df.columns.str.replace(" ", "_")

train, eval = train_test_split(
    train_df,
    train_size=0.8,
    random_state=random_state
)

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]


def objective_XGB(trial) -> float:
    params = {
        'objective': 'reg:squarederror',
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        'verbosity': 0,
        'device': 'cuda',
        'n_jobs': -1
    }

    model = XGBRegressor(**params, random_state=random_state)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)

    mape = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("MAPE: %.5f" % (mape))
    print("=" * 13)

    return mape


study_XGB = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()
)
study_XGB.optimize(objective_XGB, n_trials=opt_iter)

model_XGB = XGBRegressor(**study_XGB.best_params,
                         n_estimators=1000,
                         random_state=random_state)
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

