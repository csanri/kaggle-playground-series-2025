import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

train_df = pd.read_csv("train.csv")

train_df = train_df.drop("id", axis=1)

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

train_df = train_df.fillna("NA")

cat_cols = train_df.select_dtypes("object").columns.tolist()

colors = sns.color_palette("tab10")

fig, axes = plt.subplots(
    nrows=len(cat_cols),
    ncols=1,
    figsize=(10, 5*len(cat_cols))
)

for i, col in enumerate(cat_cols):
    sns.countplot(
        data=train_df,
        x=col,
        ax=axes[i],
        color=colors[i]
    )

plt.tight_layout()
plt.show()

