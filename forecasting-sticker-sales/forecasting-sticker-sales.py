import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "num_sold"
random_state = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

print("=" * 30)
print(train_df.info())
print("=" * 30)
print(train_df.describe())

for col in train_df:
    print("=" * 30)
    print(col)
    print(train_df[col].value_counts())

# plt.figure(figsize=(20, 16))

# sns.heatmap(train_df.isna(), cbar=False, yticklabels=False)
# plt.title("Plot of missing values")

# plt.show()

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("date")

# colors = sns.color_palette("tab10", len(cat_cols))

# fig, axes = plt.subplots(
#     nrows=3,
#     ncols=1,
#     figsize=(10 * len(cat_cols), 10)
# )

# for i, col in enumerate(cat_cols):
#     sns.countplot(
#         data=train_df,
#         x=col,
#         ax=axes[i],
#         color=colors[i]
#     )

# plt.show()

# plt.figure(figsize=(20, 16))

# sns.histplot(
#     data=train_df,
#     x=target,
#     color=colors[0],
#     kde=True
# )

# plt.show()


def date(df):
    df["date"] = pd.to_datetime(
        df["date"],
        format="ISO8601"
    )

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

    df['Group'] = (df['Year']-2020)*48 + df['Month']*4 + df['Day']//7

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

X_train_nn = np.array(X_train)
X_train_nn = torch.tensor(
    X_train_nn, dtype=torch.float32
)

X_eval_nn = np.array(X_eval)
X_eval_nn = torch.tensor(
    X_eval_nn, dtype=torch.float32
)

y_train_nn = np.array(y_train)
y_train_nn = torch.tensor(
    y_train_nn, dtype=torch.float32
).view(-1, 1)

nn_params = {
    "hidden_size": 128,
    "num_layers": 5,
    "dropout": 0.2,
    "lr": 1e-4,
    "weight_decay": 1e-5,
}


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        lr,
        weight_decay
    ):
        super(Model, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model_NN = Model(
    input_size=X_train_nn.shape[-1],
    **nn_params
)

data_loader = data.DataLoader(
    data.TensorDataset(X_train_nn, y_train_nn),
    batch_size=32
)

for epoch in range(100):
    model_NN.train()
    epoch_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        model_NN.optimizer.zero_grad()

        # Forward pass
        outputs = model_NN.forward(inputs)

        # Calculate loss
        loss = model_NN.loss_fn(outputs, targets)

        # Backward pass
        loss.backward()
        model_NN.optimizer.step()

        epoch_loss += loss.item()

    # Log the training progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(data_loader)}")

model_NN.eval()
with torch.no_grad():
    y_pred_train = model_NN(X_train_nn)
    y_pred = model_NN(X_eval_nn)

y_pred = y_pred.numpy()

score_NN = mean_absolute_percentage_error(y_true=y_eval, y_pred=y_pred)
print(score_NN)

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

