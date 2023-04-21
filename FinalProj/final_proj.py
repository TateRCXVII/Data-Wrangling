import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# read in the data for each state
ut_routes = pd.read_json("data/ut-routes.jsonlines", lines=True)
co_routes = pd.read_json("data/co-routes.jsonlines", lines=True)
id_routes = pd.read_json("data/id-routes.jsonlines", lines=True)
az_routes = pd.read_json("data/az-routes.jsonlines", lines=True)
ca_routes = pd.read_json("data/ca-routes.jsonlines", lines=True)
nv_routes = pd.read_json("data/nv-routes.jsonlines", lines=True)
nm_routes = pd.read_json("data/nm-routes.jsonlines", lines=True)

pd.set_option("display.max_rows", 5000)
pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 3000)


# create a new column in the dataframe for the type of route
# define lambda function to extract the key from the dictionary
def get_type(x):
    keys = list(x.keys())
    if len(keys) == 0:
        return "unknown"
    elif len(keys) > 1:
        return "multiple"
    else:
        return keys[0]


ut_routes["type"] = ut_routes["type"].apply(get_type)
co_routes["type"] = co_routes["type"].apply(get_type)
id_routes["type"] = id_routes["type"].apply(get_type)
az_routes["type"] = az_routes["type"].apply(get_type)
ca_routes["type"] = ca_routes["type"].apply(get_type)
nv_routes["type"] = nv_routes["type"].apply(get_type)
nm_routes["type"] = nm_routes["type"].apply(get_type)

# let's create dataframes that only include boulder routes
ut_boulder = ut_routes[ut_routes["type"] == "boulder"]
co_boulder = co_routes[co_routes["type"] == "boulder"]
id_boulder = id_routes[id_routes["type"] == "boulder"]
az_boulder = az_routes[az_routes["type"] == "boulder"]
ca_boulder = ca_routes[ca_routes["type"] == "boulder"]
nv_boulder = nv_routes[nv_routes["type"] == "boulder"]
nm_boulder = nm_routes[nm_routes["type"] == "boulder"]


# only considering the Font grade system
ut_boulder["grade"] = ut_boulder["grade"].apply(lambda x: x.get("Font", None))
co_boulder["grade"] = co_boulder["grade"].apply(lambda x: x.get("Font", None))
id_boulder["grade"] = id_boulder["grade"].apply(lambda x: x.get("Font", None))
az_boulder["grade"] = az_boulder["grade"].apply(lambda x: x.get("Font", None))
ca_boulder["grade"] = ca_boulder["grade"].apply(lambda x: x.get("Font", None))
nv_boulder["grade"] = nv_boulder["grade"].apply(lambda x: x.get("Font", None))
nm_boulder["grade"] = nm_boulder["grade"].apply(lambda x: x.get("Font", None))

# get rid of rows that have null values for grade
ut_boulder = ut_boulder[ut_boulder["grade"].notna()]
co_boulder = co_boulder[co_boulder["grade"].notna()]
id_boulder = id_boulder[id_boulder["grade"].notna()]
az_boulder = az_boulder[az_boulder["grade"].notna()]
ca_boulder = ca_boulder[ca_boulder["grade"].notna()]
nv_boulder = nv_boulder[nv_boulder["grade"].notna()]
nm_boulder = nm_boulder[nm_boulder["grade"].notna()]

# get ride of rows that have None values for grade
ut_boulder = ut_boulder[ut_boulder["grade"] != "None"]
co_boulder = co_boulder[co_boulder["grade"] != "None"]
id_boulder = id_boulder[id_boulder["grade"] != "None"]
az_boulder = az_boulder[az_boulder["grade"] != "None"]
ca_boulder = ca_boulder[ca_boulder["grade"] != "None"]
nv_boulder = nv_boulder[nv_boulder["grade"] != "None"]
nm_boulder = nm_boulder[nm_boulder["grade"] != "None"]

# get rid of ? grades
ut_boulder = ut_boulder[ut_boulder["grade"] != "?"]
co_boulder = co_boulder[co_boulder["grade"] != "?"]
id_boulder = id_boulder[id_boulder["grade"] != "?"]
az_boulder = az_boulder[az_boulder["grade"] != "?"]
ca_boulder = ca_boulder[ca_boulder["grade"] != "?"]
nv_boulder = nv_boulder[nv_boulder["grade"] != "?"]
nm_boulder = nm_boulder[nm_boulder["grade"] != "?"]


# combine the description for each route into one string
ut_boulder["description"] = ut_boulder["description"].apply(lambda x: " ".join(x))
co_boulder["description"] = co_boulder["description"].apply(lambda x: " ".join(x))
id_boulder["description"] = id_boulder["description"].apply(lambda x: " ".join(x))
az_boulder["description"] = az_boulder["description"].apply(lambda x: " ".join(x))
ca_boulder["description"] = ca_boulder["description"].apply(lambda x: " ".join(x))
nv_boulder["description"] = nv_boulder["description"].apply(lambda x: " ".join(x))
nm_boulder["description"] = nm_boulder["description"].apply(lambda x: " ".join(x))


def font_grade_to_numeric(grade):
    sub_grade_map = {"A": 1, "B": 2, "C": 3}

    if grade[-1] == "+":
        base = grade[:-1]
        modifier = 1
    elif grade[-1] == "-":
        base = grade[:-1]
        modifier = -1
    else:
        base = grade
        modifier = 0

    if len(base) > 1:
        main_grade, sub_grade = base[:-1], base[-1]
        if sub_grade in sub_grade_map:
            numeric_grade = (
                float(main_grade) * 10 + sub_grade_map[sub_grade] + modifier / 10
            )
        else:
            numeric_grade = float(main_grade) * 10 + int(sub_grade) + modifier / 10
    else:
        numeric_grade = float(base) * 10 + modifier / 10

    return numeric_grade


def sort_font_grades(grades):
    return sorted(grades, key=font_grade_to_numeric)


font_grades = [
    "3",
    "4-",
    "4",
    "4+",
    "5-",
    "5",
    "5+",
    "6A-",
    "6A",
    "6A+",
    "6B-",
    "6B",
    "6B+",
    "6C-",
    "6C",
    "6C+",
    "7A-",
    "7A",
    "7A+",
    "7B-",
    "7B",
    "7B+",
    "7C-",
    "7C",
    "7C+",
    "8A-",
    "8A",
    "8A+",
    "8B-",
    "8B",
    "8B+",
    "8C-",
    "8C",
    "8C+",
]
sorted_font_grades = sort_font_grades(font_grades)

ut_grades_count = {grade: 0 for grade in sorted_font_grades}
co_grades_count = {grade: 0 for grade in sorted_font_grades}
id_grades_count = {grade: 0 for grade in sorted_font_grades}
az_grades_count = {grade: 0 for grade in sorted_font_grades}
ca_grades_count = {grade: 0 for grade in sorted_font_grades}
nv_grades_count = {grade: 0 for grade in sorted_font_grades}
nm_grades_count = {grade: 0 for grade in sorted_font_grades}


for grade in ut_boulder["grade"]:
    ut_grades_count[grade] += 1

for grade in co_boulder["grade"]:
    co_grades_count[grade] += 1

for grade in id_boulder["grade"]:
    id_grades_count[grade] += 1

for grade in az_boulder["grade"]:
    az_grades_count[grade] += 1

for grade in ca_boulder["grade"]:
    ca_grades_count[grade] += 1

for grade in nv_boulder["grade"]:
    nv_grades_count[grade] += 1

for grade in nm_boulder["grade"]:
    nm_grades_count[grade] += 1

# plot all the states in the same histogram with different colors and opacity and labels
# Make sure all dictionaries have the same keys (grades)
all_grades = (
    set(ut_grades_count.keys())
    | set(co_grades_count.keys())
    | set(id_grades_count.keys())
    | set(az_grades_count.keys())
    | set(ca_grades_count.keys())
    | set(nv_grades_count.keys())
    | set(nm_grades_count.keys())
)

for grade in all_grades:
    if grade not in ut_grades_count:
        ut_grades_count[grade] = 0
    if grade not in co_grades_count:
        co_grades_count[grade] = 0
    if grade not in id_grades_count:
        id_grades_count[grade] = 0
    if grade not in az_grades_count:
        az_grades_count[grade] = 0
    if grade not in ca_grades_count:
        ca_grades_count[grade] = 0
    if grade not in nv_grades_count:
        nv_grades_count[grade] = 0
    if grade not in nm_grades_count:
        nm_grades_count[grade] = 0

sorted_grades = sort_font_grades(list(all_grades))

n_states = 5
index = np.arange(len(sorted_grades))
bar_width = 0.15
opacity = 0.8

plt.bar(
    index,
    [ut_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="b",
    label="Utah",
)
plt.bar(
    index + bar_width,
    [co_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="r",
    label="Colorado",
)
plt.bar(
    index + 2 * bar_width,
    [id_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="g",
    label="Idaho",
)
plt.bar(
    index + 3 * bar_width,
    [az_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="y",
    label="Arizona",
)
plt.bar(
    index + 4 * bar_width,
    [ca_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="m",
    label="California",
)
plt.bar(
    index + 5 * bar_width,
    [nv_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="c",
    label="Nevada",
)
plt.bar(
    index + 6 * bar_width,
    [nm_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="k",
    label="New Mexico",
)


plt.xlabel("Font Grades")
plt.ylabel("Counts")
plt.title("Bouldering Grade Histogram by State")
plt.xticks(index + 2 * bar_width, sorted_grades, rotation=90)
plt.legend()

plt.tight_layout()
plt.show()


plt.bar(
    index,
    [ut_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="b",
    label="Utah",
)
plt.bar(
    index + bar_width,
    [co_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="r",
    label="Colorado",
)
plt.bar(
    index + 2 * bar_width,
    [id_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="g",
    label="Idaho",
)
plt.bar(
    index + 3 * bar_width,
    [az_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="y",
    label="Arizona",
)
plt.bar(
    index + 4 * bar_width,
    [ca_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="m",
    label="California",
)
plt.bar(
    index + 5 * bar_width,
    [nv_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="c",
    label="Nevada",
)
plt.bar(
    index + 6 * bar_width,
    [nm_grades_count[grade] for grade in sorted_grades],
    bar_width,
    alpha=opacity,
    color="k",
    label="New Mexico",
)

# Find the index of the grade "8A" in the sorted_grades list
start_grade = "8A"
start_index = sorted_grades.index(start_grade)

plt.xlabel("Font Grades")
plt.ylabel("Counts")
plt.title("Bouldering Grade Histogram by State (8A and above)")
plt.xticks(
    index[start_index:] + 2 * bar_width, sorted_grades[start_index:], rotation=90
)
plt.legend()

# Set x-axis limits to zoom in on grades 8A and above
plt.xlim(start_index - 0.5, len(sorted_grades) - 0.5)
plt.ylim(0, 100)

plt.tight_layout()
plt.show()


# Download the VADER lexicon
nltk.download("vader_lexicon")

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


# Define a function to get the sentiment scores for each description
def get_sentiment_scores(text):
    return sia.polarity_scores(text)


# Calculate the average sentiment scores for each state
states = ["Utah", "Colorado", "Idaho", "Arizona", "California", "Nevada", "New Mexico"]
state_data = [
    ut_boulder,
    co_boulder,
    id_boulder,
    az_boulder,
    ca_boulder,
    nv_boulder,
    nm_boulder,
]
state_scores = []

for data in state_data:
    scores = [get_sentiment_scores(desc)["compound"] for desc in data["description"]]
    avg_score = np.mean(scores)
    state_scores.append(avg_score)

# Plot the results using a bar chart
plt.bar(states, state_scores)
plt.xlabel("States")
plt.ylabel("Average Sentiment Score")
plt.title("Sentiment Analysis of Bouldering Descriptions by State")
plt.show()

# add a new column to each dataset with the state name
ut_boulder["state"] = "Utah"
co_boulder["state"] = "Colorado"
id_boulder["state"] = "Idaho"
az_boulder["state"] = "Arizona"
ca_boulder["state"] = "California"
nv_boulder["state"] = "Nevada"
nm_boulder["state"] = "New Mexico"

# combine all the datasets into one
all_boulder = pd.concat(
    [ut_boulder, co_boulder, id_boulder, az_boulder, ca_boulder, nv_boulder, nm_boulder]
)

safety_and_stars = pd.read_csv("data/Boulder_Safety_and_Stars.csv")

# pull the mp_route_id from the metadata column in all_boulder and put it in a new column called "ID"
# metadata is a dictionary and mp_route_id is a key in that dictionary
all_boulder["ID"] = all_boulder["metadata"].apply(
    lambda x: x["mp_route_id"] if "mp_route_id" in x else None
)

# print all possible states
print(all_boulder["state"].unique())

print(all_boulder.head(20))

# drop all rows with missing ID values in all_boulder
all_boulder = all_boulder.dropna(subset=["ID"])
# drop all rows wil '' values in the ID column
all_boulder = all_boulder[all_boulder["ID"] != ""]

print(all_boulder["state"].unique())

# convert the ID column to an integer
all_boulder["ID"] = all_boulder["ID"].astype(int)
print(all_boulder["ID"].head())

# merge the all_boulder and safety_and_stars datasets on the ID column
all_boulder = all_boulder.merge(safety_and_stars, on="ID")

# plot the average rating per state
all_boulder.groupby("state")["stars"].mean().plot(kind="bar")
plt.xlabel("State")
plt.ylabel("Average Rating")
plt.title("Average Rating of Bouldering Routes by State")
plt.show()

# sentiment analysis per boulder
all_boulder["sentiment_score"] = all_boulder["description"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# Calculate the correlation coefficient between stars and sentiment_score
correlation = np.corrcoef(all_boulder["stars"], all_boulder["sentiment_score"])[0, 1]

# Create a scatter plot of stars vs sentiment_score
plt.scatter(all_boulder["stars"], all_boulder["sentiment_score"], alpha=0.5)
plt.xlabel("Stars")
plt.ylabel("Sentiment Score")
plt.title(f"Stars vs Sentiment Score (corr={correlation:.2f})")
plt.show()

# Correlation between stars and grade
all_boulder["numeric_grade"] = all_boulder["grade"].apply(font_grade_to_numeric)

# Calculate the correlation coefficient between numeric_grade and stars
correlation = np.corrcoef(all_boulder["numeric_grade"], all_boulder["stars"])[0, 1]

# Create a scatter plot of Font grades vs stars
plt.scatter(all_boulder["grade"], all_boulder["stars"], alpha=0.5)

# Label the x-axis with Font grades
plt.xticks(font_grades, rotation=90)

plt.xlabel("Font Grade")
plt.ylabel("Stars")
plt.title(f"Font Grade vs Stars (corr={correlation:.2f})")
plt.tight_layout()
plt.show()
