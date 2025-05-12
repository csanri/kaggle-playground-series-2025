import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting this so jupyter shows every column
pd.set_option('display.max_columns', None)

target = "calories"

train_df = pd.read_csv("playground-series-s5e5/train.csv")
train_df.drop("id", axis=1, inplace=True)

print("=" * 30)
print(train_df.info())
print("=" * 30)
print(train_df.describe().T)

for col in train_df.columns:
    print("-" * 30)
    print(train_df[col].value_counts())

print(train_df.isna().sum())

cat_cols = train_df.select_dtypes(include="object").columns.to_list()
num_cols = train_df.select_dtypes(exclude="object").columns.to_list()

plt.figure(figsize=(10, 18))

for col in cat_cols:
    sns.countplot(
        data=train_df,
        x=train_df[col]
    )

plt.show()

