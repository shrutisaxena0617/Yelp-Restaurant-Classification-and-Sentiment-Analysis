# Yelp Restaurants Classification and Sentiment Analysis

Using traditional machine learning and deep learning to perform sentiment analysis on customer reviews and classify restaurants into "Awesome", "Average" or "Not really!" category.

### Author's Note for the readers

This is a work in progress project which I have built from scratch starting from data cleaning to making sense of it from the business perspective and finally doing a **thorough sentiment analysis on customer reviews to classify restaurants into "Awesome", "Average" or "Not really!" category**. I have done a detailed exploratory analysis and visualizations using **plotly, bokeh, seaborn, and matplotlib**. Sentiment analysis is the heart of this project for which I have learned and applied some cool techniques such as **lemmatization** (instead of conventional tokenization or stemming approach), pos tagging etc. using **nltk** and **sentiwordnet api**. The modeling and analysis part has been done using traditional machine learning models like **SVM, Decision Tree, Gaussian Naive Bayes, Logistic Regressiona and K-Means**. Further, this project gave me a tremendous learning about how deep learning models are complex, time taking, and need a very large amount of data in order to really function as self-learning neural network. I will be updating the code to apply **Recurrent Neural Network LSTM Model using TensorFlow for sentiment analysis** and run it on the all customer reviews as this will tremendously increase the accuracy of sentiment classification and in turn make the restaurant classification much more accurate.

### About Data

Leveraged Yelp data available on Kaggle for business, checkins, reviews and tips

Data Source - https://www.kaggle.com/yelp-dataset/yelp-dataset/data

### Data Preparation

Since I was dealing with text data for sentiment analysis, I moved forward by downsizing the data to run the model on a smaller subset in the interest of time and to keep the computing complexity low. The same code can be run on AWS Sagemaker or other cloud platforms on larger dataset in significantly lesser amount of time. I ultimately joined all the different datasets to create a master dataframe and then did the analysis on it.

### Exploratory Data Analysis

Check out the notebook to explore the visuals and business insights! Some were so helpful that I used them in my thought process for feature selection as well.

### Sentiment Analysis

Wrote this code as a "learn by doing" approach and although it is not the most efficient version as of now, it does give a walkthrough of steps to perform while doing sentiment analysis for the first time. And it does give great results too!

### Machine Learning

I used scikit-learn library to try and test different classification models using a train-test split of 70-30 since I downsized the dataset. If you are running the code on a larger dataset, the training size can be increased.

### Deep Learning

I will be using TensorFlow Recurrent Neural Network LSTM model for snetiment analysis. Check out the prototype code which I built for doing log analytics using the same model - https://github.com/shrutisaxena0617/AWS_Spark_ETL_Log_Analysis/blob/master/prototype_deep_learning_RNN_LSTM.ipynb

