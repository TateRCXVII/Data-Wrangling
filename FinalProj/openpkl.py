import pandas as pd

df = pd.read_pickle(
    "data/CuratedWithRatings_OpenBetaAug2020_RytherAnderson.pkl.zip", compression="zip"
)

print(df.head(50))

# count rows
print(len(df))

# print all possible values in the location column
print(df["location"])

# print all the headers
print(df.columns)

# print parent_loc column
print(df["parent_loc"])
