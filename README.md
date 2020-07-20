# Twitter_Sentiment
PROJECT DESCRIPTION: The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset

DATASET: https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech 
The dataset have been splitted into 2 parts that are test.csv and train.csv .

METHOD : The given data sets are comprised of very much unstructured tweets which should be preprocessed to make an NLP model. In this project, we tried out the following techniques of preprocessing the raw data. But the preprocessing techniques is not limited.
Removal of punctuations.
Removal of commonly used words (stopwords).
Normalization of words.
We choose naive bayes classifier for this binary classification since it is the most common algorithm used in NLP. Moreover, we use machine learning pipeline technique which is a built-in function of scikit-learn to pre-define a workflow of algorithm. This saves much time and computational power.


RESULT:An accuracy of 0.93837 is obtained for our simple pipeline model
