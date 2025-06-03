import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

colors = sns.color_palette("tab10")

train_df = pd.read_csv("playground-series-s5e6/train.csv")
train_df = train_df.drop("id", axis=1)

target = "Fertilizer Name"

#############
# DATA INFO #
#############

print("=" * 30)
print(train_df.info())
# NOTE: The target is categotical,
# 2 categorical, and 6 numerical features

print("=" * 30)
print(train_df.describe().T)
# NOTE: Phosphorous has relatively high standard deviation
# compared to its mean

for col in train_df.columns:
    print("=" * 30)
    print(f"Values for {col}")
    print("-" * 30)
    print(train_df[col].value_counts())
# NOTE: 'Soil Type' has 5 different values
# 'Crop Type' has 11 different values
# 'Fertilizer Name' has 7 different values
# Features seem to be evenly distributed

print("=" * 30)
print(train_df.isna().sum())
# NOTE: No missing values

######################
# DATA VISUALIZATION #
######################

cat_cols = train_df.select_dtypes(include="object").columns.to_list()
num_cols = train_df.select_dtypes(exclude="object").columns.to_list()

cat_cols.remove(target)

# Distirbution of categorical features
fig, axes = plt.subplots(
    figsize=(20, 12),
    nrows=1,
    ncols=2
)

for i, col in enumerate(cat_cols):
    values = train_df[col].value_counts()
    axes[i].pie(
        x=values.values,
        labels=values.index,
        autopct="%1.1f%%",
        colors=colors
    )
    axes[i].set_title(f'Distribution of {col}', fontsize=14)

plt.tight_layout()
plt.savefig("figures/categorical_distribution.png", dpi=300)
plt.show()
# NOTE: The categorical data is evenly distributed

# Numerical distribution
fig, axes = plt.subplots(
    figsize=(20, 12),
    nrows=2,
    ncols=3
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
plt.savefig("figures/numerical_distribution.png", dpi=300)
plt.show()
# NOTE: The numerical data is evenly distributed

# Target distribution
plt.figure(figsize=(20, 20))

sns.countplot(
    data=train_df,
    x=target
)

plt.tight_layout()
plt.savefig("figures/target_distribution.png", dpi=300)
plt.show()
# NOTE: The numerical data is evenly distributed

# Pair Plot of Numerical data depending on the target
plt.figure(figsize=(20, 20))

sns.pairplot(
    data=train_df,
)

plt.tight_layout()
plt.savefig("figures/pair_plot.png", dpi=300)
plt.show()


