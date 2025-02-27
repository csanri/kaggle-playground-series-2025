import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

train_df = pd.read_csv("train.csv")
train_df = train_df.drop("id", axis=1)

target = "Price"
random_state = 42

train_df.columns = train_df.columns.str.replace(" ", "_")

print(train_df.info())
print("=" * 30)
print(train_df.describe())
print("=" * 30)

cols = train_df.columns.tolist()

for col in cols:
    print("-" * 30)
    print(train_df[col].value_counts())

print("=" * 30)
print(train_df.isna().sum())
print("=" * 30)

cat_cols = train_df.select_dtypes("object").columns.tolist()
num_cols = train_df.select_dtypes(exclude="object").columns.tolist()
num_cols.remove(target)

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

# fig, axes = plt.subplots(
#     nrows=len(num_cols),
#     ncols=2,
#     figsize=(20, 5 * len(num_cols))
# )

# for i, col in enumerate(num_cols):
#     sns.histplot(
#         data=train_df,
#         x=col,
#         kde=True,
#         bins=30,
#         ax=axes[i, 0],
#         color=colors[i]
#     )

#     sns.boxplot(
#         data=train_df,
#         x=col,
#         ax=axes[i, 1],
#         color=colors[i]
#     )

# plt.tight_layout()
# plt.savefig("images/num_data.jpeg", dpi=300)
# plt.plot()

# fig, axes = plt.subplots(
#     nrows=1,
#     ncols=2,
#     figsize=(10, 5)
# )

# sns.histplot(
#     data=train_df,
#     x=target,
#     kde=True,
#     ax=axes[0]
# )

# sns.boxplot(
#     data=train_df,
#     x=target,
#     ax=axes[1]
# )

# plt.tight_layout()
# plt.savefig("images/target.jpeg", dpi=300)
# plt.plot()

train, eval = train_test_split(
    train_df,
    train_size=0.8,
    random_state=random_state
)

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="constant", fill_value="NA")

train[num_cols] = num_imputer.fit_transform(train[num_cols])
eval[num_cols] = num_imputer.fit_transform(eval[num_cols])

for df in [train, eval]:
    df["Laptop_Compartment"] = df["Laptop_Compartment"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

    df["Waterproof"] = df["Waterproof"].map(
        {
            "No": 0,
            "Yes": 1
        }
    )

dummy_cols = ["Brand", "Material", "Size", "Style", "Color"]

train_df = pd.get_dummies(
    data=train_df,
    prefix_sep="_",
    columns=dummy_cols,
    dtype=np.int64,
)

for col in num_cols:
    print(f"Skewness of {col}")
    print(train_df[col].skew())

ss = StandardScaler()

X_train, y_train = train.drop(target, axis=1), train[target]
X_eval, y_eval = eval.drop(target, axis=1), eval[target]

X_train[num_cols] = ss.fit_transform(X_train[num_cols], y=y_train)
X_eval[num_cols] = ss.fit_transform(X_eval[num_cols], y=y_eval)

print(train_df)

