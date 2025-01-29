import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from sklearn.metrics import mean_absolute_percentage_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "num_sold"
random_state = 42
opt_iter = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

test_df = pd.read_csv("test.csv")
idx = test_df["id"]
test_df = test_df.drop("id", axis=1)

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

test_df["date"] = pd.to_datetime(
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

merged_df = pd.merge(
    weight_over_time, gdp,
    left_on=["date", "country"],
    right_on=["Year", "Country Name"],
    how="left"
)

merged_df = merged_df[["Year", "country", "GDP per Capita"]]

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

# for country in countries:
#     na_values = train_df[(train_df[target].isna()) & (train_df["country"] == country)]
#     for index, row in na_values.iterrows():
#         year = row["date"].year
#         matching = year_avg[(year_avg["country"] == country) & (year_avg["year"] == year)]
#         if not matching.empty:
#             value = matching[target].values[0]
#             train_df.loc[index, target] = value

target_mask = (
    (train_df["country"] == "Kenya") &
    (train_df["store"] == "Discount Stickers") &
    (train_df["product"] == "Holographic Goose")
)
source_mask = (
    (train_df["country"] == "Finland") &
    (train_df["store"] == "Discount Stickers") &
    (train_df["product"] == "Holographic Goose")
)

kenya_sums = train_df.loc[train_df["country"] == "Kenya"].groupby("date")["num_sold"].sum()
finland_sums = train_df.loc[train_df["country"] == "Finland"].groupby("date")["num_sold"].sum()
kenya_finland_ratio = (kenya_sums / finland_sums).mean()

train_df.loc[target_mask, "num_sold"] = train_df.loc[target_mask, "num_sold"].where(
    train_df.loc[target_mask, "num_sold"].notna(),
    train_df.loc[source_mask, "num_sold"].values * kenya_finland_ratio
)

target_mask = (
    (train_df["country"] == "Canada") &
    (train_df["store"] == "Discount Stickers") &
    (train_df["product"] == "Holographic Goose")
)
train_df.loc[target_mask, "num_sold"] = train_df.loc[target_mask, "num_sold"].where(
    train_df.loc[target_mask, "num_sold"].notna(),
    train_df.loc[source_mask, "num_sold"].values
)

masks = [
    (train_df["country"] == "Canada") &
    (train_df["store"] == "Premium Sticker Mart") &
    (train_df["product"] == "Holographic Goose"),

    (train_df["country"] == "Kenya") &
    (train_df["store"] == "Stickers for Less") &
    (train_df["product"] == "Holographic Goose"),

    (train_df["country"] == "Kenya") &
    (train_df["store"] == "Discount Stickers") &
    (train_df["product"] == "Kerneler")
]

for mask in masks:
    train_df.loc[mask, "num_sold"] = train_df.loc[mask, "num_sold"].ffill()

print(f"Missing values remaining: {train_df[target].isna().sum()}")
print(train_df[target].value_counts())

train_df = train_df[~train_df[target].isna()]


def date(df) -> pd.DataFrame:
    df["Year"] = df["date"].dt.year
    df["Day"] = df["date"].dt.day
    df["Month"] = df["date"].dt.month

    df["Year_sin"] = np.sin(2 * np.pi * df["Year"])
    df["Year_cos"] = np.cos(2 * np.pi * df["Year"])

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    # df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)

    df.drop(["Day", "Month"], axis=1, inplace=True)

    return df


def add_gdp_features(df) -> pd.DataFrame:
    df["Year"] = df["date"].dt.year

    merged = pd.merge(
        df,
        merged_df,
        on=["country", "Year"],
        how="left"
    )

    merged["GDP per Capita"] = merged.groupby(
        "country"
    )["GDP per Capita"].ffill()

    df = df.drop("Year", axis=1)

    return merged


# trian_df = add_gdp_features(train_df)


def feature_engineering(df) -> pd.DataFrame:
    df = date(df)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(
        df,
        columns=cat_cols,
        prefix_sep="_",
        drop_first=True,
        dtype="int"
    )
    df["date"] = pd.to_datetime(
        df["date"],
        format="ISO8601"
    ).astype(np.int64) / 10**9

    df.columns = df.columns.str.replace(" ", "_")
    return df


train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


# def create_data(df, seq_len):
#     sequences = []
#     targets = []

#     data = df.values

#     for i in range(len(data) - seq_len):
#         seq = data[i:i + seq_len]

#         x_seq = np.delete(seq, 1, axis=1)
#         y_seq = data[i + seq_len][1]

#         sequences.append(x_seq)
#         targets.append(y_seq)

#     x = torch.tensor(
#         np.array(sequences),
#         dtype=torch.float32
#     ).to(device)

#     y = torch.tensor(
#         np.array(targets),
#         dtype=torch.float32
#     ).to(device)

#     return x, y


split_date = train_df["date"].quantile(0.8)

train = train_df[train_df["date"] <= split_date]
eval = train_df[train_df["date"] > split_date]

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

# X_train, y_train = create_data(train, seq_len=30)
# X_eval, y_eval = create_data(eval, seq_len=30)


# class Model(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_size,
#         num_layers,
#         output_size,
#         dropout,
#         lr,
#     ):
#         super(Model, self).__init__()

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.linear = nn.Linear(hidden_size, output_size)
#         self.dropout = dropout
#         self.lr = lr
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.to(self.device)
#         self.optimizer = torch.optim.Adam(
#             params=self.parameters(),
#             lr=self.lr,
#         )

#     def forward(self, x):
#         h_0 = torch.zeros(
#             self.num_layers,
#             x.size(0),
#             self.hidden_size
#         ).to(self.device)

#         c_0 = torch.zeros(
#             self.num_layers,
#             x.size(0),
#             self.hidden_size,
#         ).to(self.device)

#         x, _ = self.lstm(x, (h_0, c_0))
#         x = self.linear(x[:, -1, :])

#         return x


# model_LSTM = Model(
#     input_size=X_train.size(2),
#     hidden_size=10,
#     num_layers=5,
#     output_size=1,
#     dropout=0.2,
#     lr=0.01
# )

# loss_fn = nn.MSELoss()
# loader = data.DataLoader(
#     dataset=data.TensorDataset(X_train, y_train),
#     shuffle=True,
#     batch_size=1
# )

# epochs = 2000

# for epoch in range(epochs):
#     model_LSTM.train()
#     for X_batch, y_batch in loader:
#         y_pred = model_LSTM(X_batch)
#         loss = loss_fn(y_pred, y_batch.unsqueeze(1))
#         model_LSTM.optimizer.zero_grad()
#         loss.backward()
#         model_LSTM.optimizer.step()
#     if epoch % 100 != 0:
#         continue
#     model_LSTM.eval()
#     with torch.no_grad():
#         y_pred = model_LSTM(X_train)
#         train_rmse = np.sqrt(loss_fn(y_pred, y_train))
#         y_pred = model_LSTM(X_eval)
#         eval_rmse = np.sqrt(loss_fn(y_pred, y_eval))
#     print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, eval_rmse))


def objective_XGB(trial) -> float:
    params = {
        'objective': 'reg:squarederror',
        "n_estimators": trial.suggest_int('n_estimators', 500, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        'verbosity': 0,
        'device': 'cuda',
        'n_jobs': -1,
        "eval_metric": "mape"
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

model_XGB = XGBRegressor(
    **study_XGB.best_params,
    random_state=random_state
)
model_XGB.fit(X_train, y_train)

y_pred = model_XGB.predict(X_eval)

score_XGB = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)

print(f"XGB score: {score_XGB}")


def objective_LGBM(trial) -> float:
    params = {
        'objective': 'regression',
        "n_estimators": trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'verbosity': -1,
        'random_state': random_state,
        "metric": "mape"
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)

    mape = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)

    print("=" * 13)
    print("MAPE: %.5f" % (mape))
    print("=" * 13)

    return mape


study_LGBM = optuna.create_study(direction='minimize')
study_LGBM.optimize(objective_LGBM, n_trials=opt_iter)

model_LGBM = LGBMRegressor(
    **study_LGBM.best_params,
    random_state=random_state
)
model_LGBM.fit(X_train, y_train)

y_pred = model_LGBM.predict(X_eval)

score_LGBM = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)

print(f"LGBM score: {score_LGBM}")

y_pred_xgb = model_XGB.predict(X_eval)
y_pred_lgbm = model_LGBM.predict(X_eval)
ensemble_pred = (y_pred_xgb + y_pred_lgbm) / 2
ensemble_mape = mean_absolute_percentage_error(y_eval, ensemble_pred)

print(f"Ensemble score: {ensemble_mape}")

y_pred_xgb_test = model_XGB.predict(test_df)
y_pred_lgbm_test = model_LGBM.predict(test_df)
ensemble_pred_test = (y_pred_xgb_test + y_pred_lgbm_test) / 2

submission = pd.DataFrame({
    "id": idx,
    target: y_pred_xgb_test
})

submission.to_csv(
    f"submissions/submission_{
        datetime.now().strftime("%Y%m%d_%H%M%S")
    }.csv",
    index=False
)

