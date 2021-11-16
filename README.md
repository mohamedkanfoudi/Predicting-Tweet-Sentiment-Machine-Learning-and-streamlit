# Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit
click in Run all button  in  requirement.ipynb  (I prefere using VS Code  )

TO train and then Predict !


download
the CSV file  !
https://drive.google.com/file/d/1lcLdqyVG6mlJUom5q9BBxBdI3puRChTn/view?usp=sharing 

and save it in  Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit FILE


Install Anaconda 
https://www.anaconda.com/products/individual#windows



![image](https://user-images.githubusercontent.com/76444482/142013402-a446b389-71e6-40a6-83cf-af6c6c448f85.png)
![image](https://user-images.githubusercontent.com/76444482/142013451-b4286abf-05f3-4fe8-9482-707d00190fed.png)

in the terminal appears , run:

pip install streamlit



run SentimentPrediction.ipynb  (don't forget to paste you own link : 
here :

# paste your link to load the training_train-1M6.csv
dataset = load_dataset("C:/Users//////Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit/training_train-1M6.csv", ['target', 't_id', 'created_at', 'query', 'user', 'text'])

and  : 
test_file_name = "C:/Users//////twitter_python/training_test_1000.csv"

and  :
test_result.to_csv('C:/Users//////Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit/result_predictions_1000_tweets.csv')

and finally :
df_csv = pd.read_csv('C:/Users//////Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit/result_predictions_1000_tweets.csv')



---------------------------------------------------------------------------------------------------------
le guide d'installation  d'Anaconda: https://docs.anaconda.com/anaconda/install/windows/
set up : https://docs.anaconda.com/anaconda/navigator/getting-started/#managing-environments
---------------------------------------------------------------------------------------------------------

