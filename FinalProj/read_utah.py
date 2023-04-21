import pandas as pd

df = pd.read_csv("data/ut-ratings.csv.zip", compression="zip")

print(df.head(100))


df_usa = pd.read_json("data/openbeta-usa-routes-aug-2020.zip", lines=True)

# print(df_usa.sample(5))
print(df_usa.columns)

# # print location for df_usa
# print(df_usa["location"])

# print(df.columns)

df_ut_routes = pd.read_json(
    "data/openbeta-routes-mountains1/ut-routes.jsonlines", lines=True
)

print(df_ut_routes.sample(5))
print(df_ut_routes.columns)

df_safety = pd.read_csv("data/Boulder_Safety_and_Stars.csv")

# print(df_safety.sample(5))
print(df_safety.columns)
