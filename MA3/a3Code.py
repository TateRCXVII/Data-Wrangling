# import libraries
import requests
from scipy import stats
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# set up the url
url = 'https://utahavalanchecenter.org/avalanches'
# get the html
page = requests.get(url)
# parse the html
soup = BeautifulSoup(page.text, 'html.parser')

# get the table from the 'avalanches' response in network tab
table = soup.find('table')
# test to see if the table is there
print(table)

# get the table headers
headers = [header.text for header in table.find_all('th')]
# test to see if the headers are there
print(headers)

# get the table rows
rows = [row for row in table.find_all('tr')]
# test to see if the rows are there
print(rows)

# get the table data
data = [[td.text for td in row.find_all('td')] for row in rows]
# test to see if the data is there
print(data)

# create a dataframe from the data
df = pd.DataFrame(data, columns=headers)
# test to see if the dataframe is there
print(df)

# drop the first row of the dataframe
df = df.drop(df.index[0])

# for each url formatted like this: https://utahavalanchecenter.org/avalanches?page=1
# get the html and parse it for every page
# then add the data to the dataframe
for i in range(1, 100):
    url = 'https://utahavalanchecenter.org/avalanches?page=' + str(i)
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    table = soup.find('table')
    rows = [row for row in table.find_all('tr')]
    data = [[td.text for td in row.find_all('td')] for row in rows]
    df2 = pd.DataFrame(data, columns=headers)
    df2 = df2.drop(df2.index[0])
    # use concat to add the new data to the dataframe
    df = pd.concat([df, df2])

df = df.rename(columns={df.columns[1]: 'Area'})

# clean the data (every cell has a newline character at the end)
df = df.replace(r' \n ', '', regex=True)
df = df.replace(r' \n', '', regex=True)
df = df.replace(r'\n ', '', regex=True)
df = df.replace(r'\n', '', regex=True)

# remove all leading and trailing whitespace from each cell
df = df.apply(lambda x: x.str.strip())

# convert the date column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

#test to see if the dataframe is there
print(df)

# save the dataframe to a csv file
# df.to_csv('avalanches.csv')

# # read the csv file into a dataframe
# df = pd.read_csv('MA3/avalanches.csv')

# remove the NaN values from widthc and depth
df = df.dropna(subset=['Width', 'Depth'])

# remove the ' and , from the width column and convert it to a float
df['Width'] = df['Width'].str.replace(',', '')
df['Width'] = df['Width'].str.replace("'", '')
df['Width'] = df['Width'].astype(float)




# print(df)

# # get the number of avalanches per region
# 1 region = df['Region'].value_counts()
# # test to see if the region data is there
# print(region)

# # plot the number of avalanches per region
# 1 region.plot(kind='bar')
# plt.show()

# # get the number of avalanches per trigger
# trigger = df['Trigger'].value_counts()

# # plot the number of avalanches per trigger
# trigger.plot(kind='bar')
# plt.show()

# # number of avalanches per area
# area = df['Area'].value_counts()

# # plot the number of avalanches per area
# area.plot(kind='bar')
# plt.show()


# EDA Checklist
# check the shape of the dataframe
print(df.shape)

# look at top and bottom
print(df.head())
print(df.tail())

# look at the data types
print(df.dtypes)

# look at the summary statistics
print(df.describe())

# print the n's
print(df.count())

# validate against external source
# https://www.deseret.com/utah/2021/2/7/22271152/police-identify-4-skiers-killed-in-avalanche-in-salt-lake-mountains
# 27 + avalanches on this date, Wilson Glade being skier triggered and 1000' wide

# print avalanche data for datetime 02/06/2021
print(df[df['Date'] == '2021-02-06'])
print(df[df['Date'] == '2021-02-06'].shape)


# put the avalanches in salt lake into their own dataframe, and the other avalanches into their own dataframe
slc = df[df['Region'] == 'Salt Lake']
not_slc = df[df['Region'] != 'Salt Lake']

# print the number of avalanches in salt lake
print(slc.shape)

# print the number of avalanches not in salt lake
print(not_slc.shape)


# find the average width and depth of avalanches in salt lake
width = slc['Width'].mean()

# print the average width and depth of avalanches in salt lake
print(width)

# find the average width and depth of avalanches not in salt lake
width = not_slc['Width'].mean()

# print the average width and depth of avalanches not in salt lake
print(width)

# plot the widths to see if they are normally distributed
# bin size = 10, x axis should span from 0 to 2000
slc['Width'].plot(kind='hist', bins=100, xlim=(0, 1550))
plt.show()

not_slc['Width'].plot(kind='hist', bins=100, xlim=(0, 1550))
plt.show()

# perform a two-sample exponential test on the widths of avalanches in salt lake and not in salt lake
# print the p value
t_stat, p_val = stats.ttest_ind(slc['Width'], not_slc['Width'])
print("t_stat:", t_stat)
print("pval:", p_val)

# In the context of a two-sample t-test performed using the "scipy.stats.ttest_ind" function, the "t_stat" value is the calculated t-statistic, and the "pval" value is the associated p-value.

# The t-statistic measures the difference between the means of the two samples in units of the standard error of the difference. A negative t-statistic indicates that the mean of the first sample is
# lower than the mean of the second sample. The magnitude of the t-statistic indicates the strength of the evidence against the null hypothesis of equal means. In this case, the calculated
# t_stat of - 13.092141471023927 indicates a strong evidence against the null hypothesis.

# The p-value represents the probability of observing a t-statistic as extreme as the one calculated if the null hypothesis of equal means is true. A p-value less than the significance
# level(commonly 0.05) indicates strong evidence against the null hypothesis. In this case, the calculated pval of 3.151546873233929e-38 is very small, indicating strong evidence against
# the null hypothesis.

# Therefore, in summary, the negative t-statistic and small p-value suggest that there is a statistically significant difference between the means of the two samples, with the mean of the
# first sample being lower than the mean of the second sample.


# find the frequency of avalanches in salt lake
slc_freq = slc['Date'].value_counts()

# print the frequency of avalanches in salt lake
print(slc_freq)

# find the frequency of avalanches not in salt lake
not_slc_freq = not_slc['Date'].value_counts()

# print the frequency of avalanches not in salt lake
print(not_slc_freq)

# t test the frequency of avalanches in salt lake and not in salt lake
# print the p value
t_stat, p_val = stats.ttest_ind(slc_freq, not_slc_freq)
print("t_stat:", t_stat)
print("pval:", p_val)
d

# In the context of a two-sample t-test, the "t_stat" value represents the calculated t-statistic, and the "pval" value represents the associated p-value.

# The t-statistic measures the difference between the means of the two samples in units of the standard error of the difference. A positive t-statistic
# indicates that the mean of the first sample is greater than the mean of the second sample. The magnitude of the t-statistic indicates the strength of
# the evidence against the null hypothesis of equal means. In this case, the calculated t_stat of 1.0370281167167579 indicates a relatively small difference
# between the means of the two samples.

# The p-value represents the probability of observing a t-statistic as extreme as the one calculated if the null hypothesis of equal means is true. A p-value
# less than the significance level (commonly 0.05) indicates strong evidence against the null hypothesis. In this case, the calculated pval of 0.29992015995363075
# is greater than 0.05, indicating that there is not enough evidence to reject the null hypothesis of equal means.

# Therefore, in summary, the positive t-statistic and relatively large p-value suggest that there is not enough evidence to conclude that the means of the two samples
#  are different, and we fail to reject the null hypothesis of equal means.

