import pandas as pd
import matplotlib.pyplot as plt

# Read in teams.csv
teams = pd.read_csv('data/Teams.csv')
batting = pd.read_csv('data/Batting.csv')

# print the first 5 rows of the dataset
print(teams.head())
print('------------------------------------------')
print(batting.head())

# print statistics for the dataset
print(teams.describe())

# count the rows of each dataset
print(teams.count())
print(batting.count())

# print all the unique team names
print(teams['teamID'].unique())

# make a scatter plot of the data
plt.scatter(teams['W'], teams['R'])
plt.xlabel('Wins')
plt.ylabel('Runs')
# add a trendline
z = np.polyfit(teams['W'], teams['R'], 1)
p = np.poly1d(z)
plt.plot(teams['W'], p(teams['W']), "r--")
plt.show()

# # make a histogram of the data
# plt.hist(teams['W'])
# plt.xlabel('Wins')
# plt.ylabel('Frequency')
# plt.show()

# plot only data from the franchID 'LAD'
# plt.scatter(teams[teams['franchID'] == 'LAD']['W'],
#             teams[teams['franchID'] == 'LAD']['R'])
# plt.xlabel('Wins')
# plt.ylabel('Runs')
# plt.show()

# average each team's batting average
batting_avg = batting.groupby('teamID')['H'].sum(
) / batting.groupby('teamID')['AB'].sum()

# average each team's wins
wins = teams.groupby('teamID')['W'].mean()

# average each team's on-base percentage (hits + walks, + hbp + sac flies) / at bats
obp = (batting.groupby('teamID')['H'].sum() + batting.groupby('teamID')['BB'].sum() + batting.groupby(
    'teamID')['HBP'].sum() + batting.groupby('teamID')['SF'].sum()) / batting.groupby('teamID')['AB'].sum()

slg = (batting.groupby('teamID')['H'].sum() + batting.groupby('teamID')['2B'].sum() * 2 + batting.groupby(
    'teamID')['3B'].sum() * 3 + batting.groupby('teamID')['HR'].sum() * 4) / batting.groupby('teamID')['AB'].sum()

ops = obp + slg

# make a trendline for ops
z = np.polyfit(wins, ops, 1)
p = np.poly1d(z)

# plot the ops against each team's wins
plt.scatter(wins, ops)
plt.xlabel('Wins')
plt.ylabel('OPS')

# label each bullet point with the team name only when the mouse hovers over the point
for team, x, y in zip(batting_avg.index, wins, ops):
    label = team
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')


plt.show()


# plot the batting average against each team's wins
plt.scatter(wins, batting_avg)
plt.xlabel('Wins')
plt.ylabel('Batting Average')
# label each bullet
for team, x, y in zip(batting_avg.index, wins, batting_avg):
    label = team
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()
