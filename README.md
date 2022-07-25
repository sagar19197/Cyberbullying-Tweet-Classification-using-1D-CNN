# CyberBullying_Sentiment_Analysis
Classifying the Cyber Bullying Tweets and done sentimental Analysis to classify tweets into 6 classes (Religion,Gender,Ethinicity,Other CyberBullying,Age ,Not Bullying).
## Problem Statement
Given a tweet, we are classifying the tweet as not_bullying or bullying (religion, age, ethnicity,
gender) using machine learning and natural language processing.
## Dataset -
For this problem we are using Kaggle dataset.
Dataset- https://www.kaggle.com/andrewmvd/cyberbullying-classification
## EDA and Pre-Processing
Following steps were performed for EDA & pre-processing:
● Histogram plots were plotted to check for class imbalance
● Data cleaning was performed using various libraries like nltk, re, emoji, string
● In data cleaning we removed special characters (hashtags, URLs, etc), emojis, word
contractions and multiple places between words.
● After cleaning we removed duplicate tweets
● After removing duplicate tweets, other_cyberbullying class was creating imbalance, so
this class was dropped.
● Stemming, Lemmatization was performed and their performance was compared .
● Feature extraction was done using various word embedding algorithms like TF-IDF,
Word2vec, Bag of words and GloVe and they were used to create word embeddings to
train Naive Bayes model.
## CNN 1D- 
We have implemented a One dimensional
Convolution network for the four word embeddings -
Bag of words, Tf-idf, GloVe and Word to vector. The
network takes 500 dimensional word embeddings as
input. This 500 dimensional input is then passed to
the first convolution layer with filter size as 2 and 32
feature maps. We have used padding to ensure boundary conditions and accurate analysis. Followed by this
we have a max pooling layer which will help to reduce the size of feature maps, enhancing speed and
help in reducing overfitting. Similar to this we have
added 4 more layers which contain convolution followed by max pooling. After the end of layer eight (including input layer) we flatten the layer which contains
128 neurons. Before passing these 128 neurons to the
dense layer we used dropout rate to be 0.25 to reduce
overfitting. Followed by this dropout we have another
dense layer of 32 neurons. Finally, by again using the
dropout rate to be 0.25 we reach the output layer. We
have come up with this architecture by trying out multiple architectures, changing filter sizes and changing
max pooling layer sizes. We have used the ReLu activation function between the layers. We choose this
activation function over sigmoid or tanh because of
its fast speed and resistance towards vanishing gradient..For loss function we have chosen categorical cross
entropy loss function .Also,our last layer uses softmax
as activation function. And this softmax activation
function will prevent loss (cross-entropy loss) to shoot
up to nan values.
For the output layer, we will get the probability (output
from softmax) for each of the six classes. These probabilities represent how much the given tweet is likely
to belong to a particular class. To get the output (predicted) label, we are using argmax over these probabilities.
