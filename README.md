# Sentiment Analysis using Text Mining and Natural Language Processing 

## Project Overview
This project implements sentiment analysis on Twitter data using various machine learning models. The goal is to classify tweets into positive, negative, or neutral sentiments. We employed multiple algorithms, including Logistic Regression, Naive Bayes, Support Vector Machines (SVM), and Boosting Algorithms such as Gradient Boosting, XGBoost, and AdaBoost, to determine the most effective approach for sentiment classification.

## Technical Skills and Tools
- **Machine Learning Models:**
  - **Logistic Regression:** Provides a robust method for binary and multi-class classification.
  - **Naive Bayes:** A probabilistic classifier based on Bayes' theorem with strong independence assumptions.
  - **Support Vector Machines (SVM):** Effective for high-dimensional spaces and text classification.
  - **Gradient Boosting, XGBoost, AdaBoost:** Advanced boosting techniques that improve predictive performance by combining multiple models.
  
- **Data Preprocessing:**
  - **Tokenization:** Splitting text into individual words or tokens.
  - **Stemming:** Reducing words to their root form to consolidate variations.
  - **Stop Words Removal:** Eliminating common words that do not contribute to sentiment analysis.

- **Libraries Used:**
  - **NLTK (Natural Language Toolkit):** For text preprocessing and tokenization.
  - **Scikit-learn:** For implementing machine learning models and evaluation metrics.
  - **Pandas:** For data manipulation and analysis.
  - **NumPy:** For numerical operations.

## Results
- **Best Performing Model:** Logistic Regression
- **Accuracy:** 80%
- **Performance Metrics:** Logistic Regression outperformed other models with an accuracy rate of 80%. It demonstrated the ability to effectively classify sentiment in tweets, while other models such as SVM and Boosting Algorithms showed slightly lower accuracy but still provided valuable insights.

## Team Members
- **Yaswanth Ganapathi**
- **Shirisha Gajjela**
- **Rahul Sajith**

## Dataset
The project uses a Twitter dataset obtained from Kaggle with 27,482 tweets. Each tweet is labeled with sentiments: positive, negative, or neutral.
