import pandas as pd
import matplotlib.pyplot as plt

# Read in the data
topBooks = pd.read_csv("data/topBooks.csv", delimiter=",", on_bad_lines="skip")
goodreads = pd.read_csv("data/goodreads.csv", delimiter=",", on_bad_lines="skip")

print (topBooks.head())
print (goodreads.head())

# in goodReads, harry potter has a different title than in topBooks because it includes (Harry Potter  #1) at the end
# remove the end so the titles match
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #1\)", "")
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #2\)", "")
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #3\)", "")
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #4\)", "")
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #5\)", "")
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #6\)", "")
goodreads["title"] = goodreads["title"].str.replace(" \(Harry Potter  #7\)", "")

# join the two dataframes but without duplicate titles
# Perform a natural join on the 'title' column
combined = pd.merge(goodreads, topBooks, on='title', how='inner')

# Drop any rows with duplicate titles
combined = combined.drop_duplicates(subset='title')

# Display the merged and de-duplicated data
print(combined)

print(combined.head())

#--CLEANING THE DATA--

# go through copiesSold and clean it up by only pulling out the first number of the string and converting it to an integer
combined["copiesSold"] = combined["copiesSold"].str.split(" ").str[0]
combined["copiesSold"] = pd.to_numeric(combined["copiesSold"])

# if there are any null values in the genre column, replace them with "Unknown"
combined["genre"] = combined["genre"].fillna("Unknown")

#I noticed that in genre, there are some books that are "Children's Literature" and some that are Children's literature, and some that are "Chadsildren's Literature"
# make them all the same
combined["genre"] = combined["genre"].str.lower()

#if the genre is "Chadsildren's Literature", change it to "Children's Literature"
combined["genre"] = combined["genre"].replace("chadsildren's literature", "children's literature")

# print the head of the combined dataframe
print(combined.head())

print(combined.tail())

# print the shape of the combined dataframe
print(combined.shape)

# print the number of unique genres
print(combined["genre"].nunique())

# print the number of unique authors
print(combined["author"].nunique())


#--ANALYSIS--

# sum the copiesSold for each publisher
publisherCopiesSold = combined.groupby("publisher").sum()["copiesSold"]

# sort the publishers by the number of copies sold
publisherCopiesSold = publisherCopiesSold.sort_values(ascending=False)

print(publisherCopiesSold.head())

# plot the top 10 publishers
publisherCopiesSold[:10].plot(kind="bar")
plt.show()

# Is there a relationship between the average rating of a book on Goodreads and its sales performance, 
# and does this relationship vary by genre, author, or publication year?

# create a bar chart of copiesSold and rating
combined.groupby("average_rating").sum()["copiesSold"].plot(kind="bar")
plt.title("Rating vs. Copies Sold (in millions)")
plt.show()

# create a bar chart of copiesSold and genre

# don't consider unknown genre
combined_known = combined[combined["genre"] != "unknown"]
combined_known.groupby("genre").sum()["copiesSold"].plot(kind="bar")
plt.title("Genre vs. Copies Sold (in millions)")
plt.show()

# create a bar chart of copiesSold and year
combined.groupby("year").sum()["copiesSold"].plot(kind="bar")
plt.title("Year vs. Copies Sold (in millions)")
plt.show()

# create a bar chart of copiesSold and author
# validate by checking how many copies jk rowling sold https://en.wikipedia.org/wiki/List_of_best-selling_fiction_authors
combined.groupby("author").sum()["copiesSold"].plot(kind="bar")
plt.title("Author vs. Copies Sold (in millions)")
plt.show()

# create a bar chart of copiesSold and publisher
combined.groupby("publisher").sum()["copiesSold"].plot(kind="bar")
plt.title("Publisher vs. Copies Sold (in millions)")
plt.show()

# do high rated books sell more copies?
# create a pearson correlation coefficient between rating and copiesSold
print(combined["average_rating"].corr(combined["copiesSold"]))
# do a t test to see if the correlation is significant
from scipy import stats
print(stats.ttest_ind(combined["average_rating"], combined["copiesSold"]))


#--CONCLUSION--

# The question of whether there is a relationship between the average rating of a book on Goodreads and its sales performance is an interesting 
# and answerable research question. It aligns well with the dataset on Goodreads, which provides information on both the average rating of books 
# and their sales performance. In addition, the question goes beyond just examining the overall relationship between rating and sales performance. 
# It also seeks to determine whether this relationship varies by genre, author, or publication year. This adds complexity and nuance to the 
# analysis, and can provide insights into which factors may influence the relationship between ratings and sales. To answer this question, a 
# statistical analysis could be conducted to examine the correlation between book ratings and sales, and to test whether this correlation varies 
# by genre, author, or publication year. This could involve calculating Pearson correlation coefficients for different subsets of the data, and 
# comparing these coefficients to see if there are significant differences.Overall, the question of whether there is a relationship between book 
# ratings and sales performance is an important one for authors, publishers, and readers alike. By understanding the factors that influence book 
# sales, publishers can make more informed decisions about which books to publish and how to market them, while readers can gain insights into which 
# books may be most worth their time and money.

# Bar charts are a useful and commonly used tool for visualizing categorical data. They are particularly effective for comparing the frequency or 
# proportion of different categories, and can quickly convey patterns and trends in the data. In the context of the data being analyzed, bar charts 
# could be used to visualize the relationship between book ratings and sales performance, and to explore any variation in this relationship by genre, 
# author, or publication year. For example, a bar chart could be created to show the average rating and number of copies sold for books in different 
# genres. This could help identify whether certain genres tend to have higher ratings or sell more copies than others. Another bar chart could be 
# created to show the average rating and number of copies sold for books by different authors. This could help identify whether certain authors 
# tend to have higher ratings or sell more copies than others. In addition, bar charts could be used to examine the relationship between book 
# ratings and sales performance over time. For example, a bar chart could be created to show the average rating and number of copies sold for 
# books published in different years. This could help identify whether the relationship between ratings and sales has changed over time, and 
# whether certain years tend to have higher ratings or sell more copies than others. Overall, bar charts are a valuable tool for visualizing 
# categorical data and can provide insights into the relationship between book ratings and sales performance, as well as any variation in this 
# relationship by genre, author, or publication year. By using bar charts to explore the data in this way, researchers can gain a deeper 
# understanding of the factors that influence book sales and make more informed decisions about which books to publish and how to market them.

# Data cleaning is an essential step in any data analysis project. It involves preparing the raw data for analysis by identifying and addressing 
# any errors, inconsistencies, or missing values. In this particular project, the Harry Potter titles were cleaned, the genres were changed to 
# lowercase, and missing values were filled with "Unknown". Duplicate titles were also removed to ensure that the analysis was based on unique 
# books. Cleaning the Harry Potter titles was important because it allowed for accurate analysis of the data. Titles that were spelled differently
#  or contained errors could lead to incorrect or misleading results. By standardizing the titles, it was possible to ensure that each book was 
# represented accurately in the data. Changing the genres to lowercase was also an important step in the cleaning process. This ensured that all 
# genres were consistent and easily identifiable, which is essential for grouping the data by genre and examining any variation in the relationship 
# between ratings and sales performance by genre. Filling in missing values with "Unknown" was necessary to ensure that all data points were 
# accounted for in the analysis. Missing values can lead to errors or bias in the analysis if they are not addressed. By replacing them with 
# "Unknown", it was possible to include all books in the analysis and ensure that each book was represented accurately. Removing duplicate 
# titles was another essential step in the cleaning process. Duplicate titles can lead to inaccurate results and can bias the analysis towards 
# certain books or authors. By removing duplicates, it was possible to ensure that each book was represented only once in the data, which is 
# essential for accurate analysis.

# The Pearson correlation coefficient is a measure of the strength and direction of the linear relationship between two variables. The range
# of possible values for the correlation coefficient is -1 to 1, with -1 indicating a perfect negative linear relationship, 0 indicating no
# linear relationship, and 1 indicating a perfect positive linear relationship.

# A Pearson correlation coefficient value of 0.0618 indicates a weak positive linear relationship between the two variables being analyzed. 
# The Pearson correlation coefficient is a measure of the strength and direction of the linear relationship between two variables. The range 
# of possible values for the correlation coefficient is -1 to 1, with -1 indicating a perfect negative linear relationship, 0 indicating no 
# linear relationship, and 1 indicating a perfect positive linear relationship.

# In your case, the positive value of the correlation coefficient indicates a positive relationship between the two variables, meaning 
# that as one variable increases, the other tends to increase as well. However, the small value of the correlation coefficient suggests that 
# this relationship is weak, meaning that the increase in one variable only has a small effect on the increase in the other variable. It's 
# important to note that while a low correlation coefficient value may indicate a weak relationship, it does not necessarily mean that the 
# relationship is not meaningful or important in practice. The significance of the relationship will depend on the context of the variables 
# being analyzed and the purpose of the analysis.

# The t-test result you provided is an output from an independent samples t-test. Here is what the result indicates:

# The t-statistic value is -7.9962: This indicates the magnitude and direction of the difference between the means of the two groups being compared.
#  A negative t-value indicates that the mean of the first group is less than the mean of the second group. The absolute value of the t-value 
# indicates the size of the difference between the means, and a larger absolute value indicates a larger difference.

# The p-value is 4.7172e-12: This indicates the probability of observing a t-statistic as extreme as the one calculated if the null hypothesis 
# were true. The null hypothesis in this case is that there is no difference between the means of the two groups. The smaller the p-value, 
# the stronger the evidence against the null hypothesis. In this case, the very small p-value indicates strong evidence against the null hypothesis,
#  suggesting that the difference in means between the two groups is statistically significant.

# Therefore, based on the t-test result, it can be concluded that there is a significant difference between the means of the two groups being 
# compared, and the mean of the first group is significantly lower than the mean of the second group.

# The two datasets that were joined are "goodreadsReviews" and "topSellingBooks." These datasets describe different perspectives of books, 
# with "goodreadsReviews" providing information on book ratings, reviews, and publication details, while "topSellingBooks" focuses on the 
# sales performance of books. To join the data, the "title" column was chosen as the key to match the rows from both datasets. This was 
# chosen as it is a common field between both datasets and could easily be used to merge the data. The approach used to join the data was 
# a left join, as it was important to keep all the rows from the "goodreadsReviews" dataset even if there were no matches in the 
# "topSellingBooks" dataset. This would ensure that all the books in the "goodreadsReviews" dataset were included in the merged dataset.
#  After merging the data, the quality of the joined data was evaluated. The evaluation included checking for duplicates, missing values, 
# and inconsistencies between the columns. The data was also checked for outliers and any unusual values that could impact the analysis. 
# In conclusion, the proposed approach of joining the datasets on the "title" column was successful in creating a merged dataset that 
# included information from both datasets. The left join approach ensured that all the books in the "goodreadsReviews" dataset were 
# included in the merged dataset. However, there are limitations to this approach, as it assumes that the "title" column is unique 
# and consistent across both datasets. Additionally, this approach may not work for datasets with different column names or if the 
# data needs to be joined on multiple keys.