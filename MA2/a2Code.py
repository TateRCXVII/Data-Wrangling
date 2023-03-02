import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# What music do college students enjoy?
# What is the relationship between music and age?
# What is the relationship between music and Life Struggles?
# read in data/responses.csv
responses = pd.read_csv('MA2/data/responses.csv')

print(responses.head())
print(responses.tail())

print(responses.count())

print(responses.describe())


# filter these columns into a new df:
# Music Slow songs or fast songs	Dance	Folk	Country	 Classical music	Musical	 Pop	Rock	Metal or Hardrock	 Punk
# Hiphop, Rap	Reggae, Ska	Swing, Jazz	Rock n roll	Alternative	Latino	Techno, Trance	Opera
# Age	Life struggles
music_responses = responses[['Music', 'Slow songs or fast songs', 'Dance', 'Folk', 'Country', 'Classical music', 'Musical', 'Pop', 'Rock', 'Metal or Hardrock',
                             'Punk', 'Hiphop, Rap', 'Reggae, Ska', 'Swing, Jazz', 'Rock n roll', 'Alternative', 'Latino', 'Techno, Trance', 'Opera', 'Age', 'Life struggles']]

print('music responses checklist:')
print('Head:')
print(music_responses.head())
print('Tail:')
print(music_responses.tail())
print('Count:')
print(music_responses.describe())


# let's go ahead and remove the students who don't listen to music (with a rating of 1-2 in the music column)
# the students who don't listen to music will skew the data, so let's remove them. They are not relevant to our analysis.
music_responses = music_responses[music_responses['Music'] > 2]
# print the count of the music responses
print(music_responses.describe())

# filter out rows with missing data
# looks like there is a lot of missing data in the music columns and inconsistency between each column.
# let's see if dropping the data leaves us with enough data to work with
dropped_music_responses = music_responses.dropna()

print(dropped_music_responses.describe())
# all columns now have 921 rows, which is a good consistent number.
# we don't need to fill the rows with missing data because we have enough sample data to work with.

# sometimes it's worth it to fill the missing data with the mean of the column, so lets try it
filled_music_responses = music_responses.fillna(music_responses.mean())

print(filled_music_responses.describe())


# let's see how these two dataframes compare and answer the questions
# What music do college students enjoy?
# What is the relationship between music and age?
# What is the relationship between music and Life Struggles?

# # What music do college students enjoy?
# # let's see what the most popular music genres are
# # find the mean of each column, return the column with the highest mean
# # fill an array with the mean values
# dropped_music_means = []
# dropped_music_column_names = []
# for column in dropped_music_responses:
#     if column == 'Age' or column == 'Life struggles' or column == 'Music':
#         continue
#     dropped_music_means.append(dropped_music_responses[column].mean())
#     dropped_music_column_names.append(column)

# filled_music_means = []
# filled_music_column_names = []
# for column in filled_music_responses:
#     # exlude age, life struggles, and music
#     if column == 'Age' or column == 'Life struggles' or column == 'Music':
#         continue
#     filled_music_means.append(filled_music_responses[column].mean())
#     filled_music_column_names.append(column)

# # return the column with the highest mean
# dropped_max_mean = max(dropped_music_means)
# dropped_max_mean_index = dropped_music_means.index(dropped_max_mean)

# filled_max_mean = max(filled_music_means)
# filled_max_mean_index = filled_music_means.index(filled_max_mean)

# # print the column with the highest mean
# print('Dropped Music Responses:')
# print(dropped_music_responses.columns[dropped_max_mean_index])
# print('Filled Music Responses:')
# print(filled_music_responses.columns[filled_max_mean_index])

dropped_music_means = dropped_music_responses.mean()
dropped_music_means = dropped_music_means.drop(
    ['Age', 'Life struggles', 'Music'])

filled_music_means = filled_music_responses.mean()
filled_music_means = filled_music_means.drop(
    ['Age', 'Life struggles', 'Music'])

# make an array of the column names
music_column_names = []
for column in dropped_music_responses:
    if column == 'Age' or column == 'Life struggles' or column == 'Music':
        continue
    music_column_names.append(column)

# add up each genre and display the total of each one
for column in dropped_music_responses:
    if column == 'Age' or column == 'Life struggles' or column == 'Music':
        continue
    print(column + ': ' + str(dropped_music_responses[column].sum()))

# make a bar chart of the mean values
plt.bar(music_column_names, dropped_music_means)
plt.title('Dropped Music Responses')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

plt.bar(music_column_names, filled_music_means)
plt.title('Filled Music Responses')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

# What is the relationship between music and age?
# categorize the age column into 3 groups: 15-19, 20-23, 24+
pd.cut(dropped_music_responses['Age'], bins=[15, 20, 24, 35], labels=[
       '15-19', '20-23', '24-35'], include_lowest=True)
print(pd.cut(filled_music_responses['Age'], bins=[15, 20, 24, 35], labels=[
    '15-19', '20-23', '24-35'], include_lowest=True))

# find the mean of each age group for each music genre

# 15-19
dropped_15_19 = dropped_music_responses[dropped_music_responses['Age'] < 20]
dropped_15_19_means = dropped_15_19.mean()
dropped_15_19_means = dropped_15_19_means.drop(
    ['Age', 'Life struggles', 'Music'])

filled_15_19 = filled_music_responses[filled_music_responses['Age'] < 20]
filled_15_19_means = filled_15_19.mean()
filled_15_19_means = filled_15_19_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 20-23
dropped_20_23 = dropped_music_responses[dropped_music_responses['Age'] < 24]
dropped_20_23 = dropped_20_23[dropped_20_23['Age'] >= 20]
dropped_20_23_means = dropped_20_23.mean()
dropped_20_23_means = dropped_20_23_means.drop(
    ['Age', 'Life struggles', 'Music'])

filled_20_23 = filled_music_responses[filled_music_responses['Age'] < 24]
filled_20_23 = filled_20_23[filled_20_23['Age'] >= 20]
filled_20_23_means = filled_20_23.mean()
filled_20_23_means = filled_20_23_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 24+
dropped_24_plus = dropped_music_responses[dropped_music_responses['Age'] >= 24]
dropped_24_plus_means = dropped_24_plus.mean()
dropped_24_plus_means = dropped_24_plus_means.drop(
    ['Age', 'Life struggles', 'Music'])

filled_24_plus = filled_music_responses[filled_music_responses['Age'] >= 24]
filled_24_plus_means = filled_24_plus.mean()
filled_24_plus_means = filled_24_plus_means.drop(
    ['Age', 'Life struggles', 'Music'])

# make a bar chart of the mean values
plt.bar(music_column_names, dropped_15_19_means)
plt.title('Dropped Music Responses: 15-19')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

plt.bar(music_column_names, filled_15_19_means)
plt.title('Filled Music Responses: 15-19')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

plt.bar(music_column_names, dropped_20_23_means)
plt.title('Dropped Music Responses: 20-23')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

plt.bar(music_column_names, filled_20_23_means)
plt.title('Filled Music Responses: 20-23')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

plt.bar(music_column_names, dropped_24_plus_means)
plt.title('Dropped Music Responses: 24+')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

plt.bar(music_column_names, filled_24_plus_means)
plt.title('Filled Music Responses: 24+')
plt.xlabel('Music Genre')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()

#  find the mean for each individual age for each music genre
# 15
dropped_15 = dropped_music_responses[dropped_music_responses['Age'] == 15]
dropped_15_means = dropped_15.mean()
dropped_15_means = dropped_15_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 16
dropped_16 = dropped_music_responses[dropped_music_responses['Age'] == 16]
dropped_16_means = dropped_16.mean()
dropped_16_means = dropped_16_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 17
dropped_17 = dropped_music_responses[dropped_music_responses['Age'] == 17]
dropped_17_means = dropped_17.mean()
dropped_17_means = dropped_17_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 18
dropped_18 = dropped_music_responses[dropped_music_responses['Age'] == 18]
dropped_18_means = dropped_18.mean()
dropped_18_means = dropped_18_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 19
dropped_19 = dropped_music_responses[dropped_music_responses['Age'] == 19]
dropped_19_means = dropped_19.mean()
dropped_19_means = dropped_19_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 20
dropped_20 = dropped_music_responses[dropped_music_responses['Age'] == 20]
dropped_20_means = dropped_20.mean()
dropped_20_means = dropped_20_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 21
dropped_21 = dropped_music_responses[dropped_music_responses['Age'] == 21]
dropped_21_means = dropped_21.mean()
dropped_21_means = dropped_21_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 22
dropped_22 = dropped_music_responses[dropped_music_responses['Age'] == 22]
dropped_22_means = dropped_22.mean()
dropped_22_means = dropped_22_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 23
dropped_23 = dropped_music_responses[dropped_music_responses['Age'] == 23]
dropped_23_means = dropped_23.mean()
dropped_23_means = dropped_23_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 24
dropped_24 = dropped_music_responses[dropped_music_responses['Age'] == 24]
dropped_24_means = dropped_24.mean()
dropped_24_means = dropped_24_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 25
dropped_25 = dropped_music_responses[dropped_music_responses['Age'] == 25]
dropped_25_means = dropped_25.mean()
dropped_25_means = dropped_25_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 26
dropped_26 = dropped_music_responses[dropped_music_responses['Age'] == 26]
dropped_26_means = dropped_26.mean()
dropped_26_means = dropped_26_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 27
dropped_27 = dropped_music_responses[dropped_music_responses['Age'] == 27]
dropped_27_means = dropped_27.mean()
dropped_27_means = dropped_27_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 28
dropped_28 = dropped_music_responses[dropped_music_responses['Age'] == 28]
dropped_28_means = dropped_28.mean()
dropped_28_means = dropped_28_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 29
dropped_29 = dropped_music_responses[dropped_music_responses['Age'] == 29]
dropped_29_means = dropped_29.mean()
dropped_29_means = dropped_29_means.drop(
    ['Age', 'Life struggles', 'Music'])

# 30
dropped_30 = dropped_music_responses[dropped_music_responses['Age'] == 30]
dropped_30_means = dropped_30.mean()
dropped_30_means = dropped_30_means.drop(
    ['Age', 'Life struggles', 'Music'])


# print all the means
print('Dropped Music Responses: 15')
print(dropped_15_means)
print('Dropped Music Responses: 16')
print(dropped_16_means)
print('Dropped Music Responses: 17')
print(dropped_17_means)
print('Dropped Music Responses: 18')
print(dropped_18_means)
print('Dropped Music Responses: 19')
print(dropped_19_means)
print('Dropped Music Responses: 20')
print(dropped_20_means)
print('Dropped Music Responses: 21')
print(dropped_21_means)
print('Dropped Music Responses: 22')
print(dropped_22_means)
print('Dropped Music Responses: 23')
print(dropped_23_means)
print('Dropped Music Responses: 24')
print(dropped_24_means)
print('Dropped Music Responses: 25')
print(dropped_25_means)
print('Dropped Music Responses: 26')
print(dropped_26_means)
print('Dropped Music Responses: 27')
print(dropped_27_means)
print('Dropped Music Responses: 28')
print(dropped_28_means)
print('Dropped Music Responses: 29')
print(dropped_29_means)
print('Dropped Music Responses: 30')
print(dropped_30_means)

# compare the means of the music genres for each age
# 15
dropped_15_means.plot(kind='bar', title='Dropped Music Responses: 15')
plt.show()
# 16
dropped_16_means.plot(kind='bar', title='Dropped Music Responses: 16')
plt.show()
# 17
dropped_17_means.plot(kind='bar', title='Dropped Music Responses: 17')
plt.show()
# 18
dropped_18_means.plot(kind='bar', title='Dropped Music Responses: 18')
plt.show()
# 19
dropped_19_means.plot(kind='bar', title='Dropped Music Responses: 19')
plt.show()
# 20
dropped_20_means.plot(kind='bar', title='Dropped Music Responses: 20')
plt.show()
# 21
dropped_21_means.plot(kind='bar', title='Dropped Music Responses: 21')
plt.show()
# 22
dropped_22_means.plot(kind='bar', title='Dropped Music Responses: 22')
plt.show()
# 23
dropped_23_means.plot(kind='bar', title='Dropped Music Responses: 23')
plt.show()
# 24
dropped_24_means.plot(kind='bar', title='Dropped Music Responses: 24')
plt.show()
# 25
dropped_25_means.plot(kind='bar', title='Dropped Music Responses: 25')
plt.show()
# 26
dropped_26_means.plot(kind='bar', title='Dropped Music Responses: 26')
plt.show()
# 27
dropped_27_means.plot(kind='bar', title='Dropped Music Responses: 27')
plt.show()
# 28
dropped_28_means.plot(kind='bar', title='Dropped Music Responses: 28')
plt.show()
# 29
dropped_29_means.plot(kind='bar', title='Dropped Music Responses: 29')
plt.show()
# 30
dropped_30_means.plot(kind='bar', title='Dropped Music Responses: 30')
plt.show()

# for each age group, see if there is a correlation between the music genres and life struggles
# 15
dropped_15.corr()
# 16
dropped_16.corr()
# 17
dropped_17.corr()
# 18
dropped_18.corr()
# 19
dropped_19.corr()
# 20
dropped_20.corr()
# 21
dropped_21.corr()
# 22
dropped_22.corr()
# 23
dropped_23.corr()
# 24
dropped_24.corr()
# 25
dropped_25.corr()
# 26
dropped_26.corr()
# 27
dropped_27.corr()
# 28
dropped_28.corr()
# 29
dropped_29.corr()
# 30
dropped_30.corr()

# let's pick a few groups and see if there is a correlation between the music genres and life struggles
# 15
dropped_15.corr().style.background_gradient(cmap='coolwarm')
# 18
dropped_18.corr().style.background_gradient(cmap='coolwarm')
# 21
dropped_21.corr().style.background_gradient(cmap='coolwarm')
# 24+
dropped_24_plus.corr().style.background_gradient(cmap='coolwarm')

# calculate the correlation for the entire population
dropped_music_responses.corr().style.background_gradient(cmap='coolwarm')
