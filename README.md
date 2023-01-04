# craigslist-electronics-enhancement
Eliminating spam and classifying electronic listings based on the descriptions on Craigslist website:


![image](https://user-images.githubusercontent.com/83170577/210628646-239dec07-24c3-4467-8a93-aa9e439f52e9.png)

For the project, we scraped the Chicago Electronics category product listings from the Craigslist
website (https://chicago.craigslist.org/search/ela).

The scraped file consisted of product title, URL link and description. For training our model, we
required a labelled sample of sizeable amount (3000 in this case). The labelling was done using
two methods- a) Coded labelling through pattern identification b) Manual labelling.


In our modelling solution, we have 3 steps:
a. Data Pre-processing: Standard process: Lemmatization and Stop word removal
b. Vectorization: We chose TF-IDF vectorization over CountVectorizer since we
wanted to highlight the importance of terms within each document (in this case
description).
c. Traditional data mining methods: We used classification models like Naïve Bayes,
Random Forest, Linear and Polynomial SVM and XG Boost for model training.
Due to high class imbalance, we tuned our hyperparameters to classify minority
class better. In the XGBoost model, we used weighted scale_pos_weight hyperparameter adjustment. We also tried SMOTE oversampling to improve
results, and a stratified Train-test split ration of 80:20. Our final model, which gave
the best recall output was Random Forest with class weighting.


