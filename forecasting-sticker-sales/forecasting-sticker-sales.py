import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "num_sold"

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

plt.figure(figsize=(20, 16))

sns.heatmap(train_df.isna(), cbar=False, yticklabels=False)
plt.title("Plot of missing values")

plt.show()

cat_cols = train_df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("date")

colors = sns.color_palette("tab10", len(cat_cols))

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(10 * len(cat_cols), 10)
)

for i, col in enumerate(cat_cols):
    sns.countplot(
        data=train_df,
        x=col,
        ax=axes[i],
        color=colors[i]
    )

plt.show()

plt.figure(figsize=(20, 16))

sns.histplot(
    data=train_df,
    x=target,
    color=colors[0],
    kde=True
)

plt.show()

train_df = pd.get_dummies(train_df[cat_cols], prefix_sep="_", drop_first=True)

print(train_df)

