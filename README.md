# Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit
(I prefere using Visual Studio Code )
 Open the folder in VS Code
 Run the first cell   in  requirement.ipynb  
VS Code will demand to install Jupiter and python, install them  ,  after that return to requirement.ipynb  and click Run ALL (you should choose your interpreter of python )
TO train and then Predict !

type in terminal  : 

python
">>>"
import nltk 
nltk.download('stopwords')
nltk.download('punkt')




download
the CSV file  !
https://drive.google.com/file/d/1lcLdqyVG6mlJUom5q9BBxBdI3puRChTn/view?usp=sharing 

and save it in  Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit FOLDER

in the terminal , run:

pip install streamlit

______________________________________________________________
Now , you can run directly web app  because the result of training is already saved in saved_steps1.pkl   using pickle  
and the result of predicting is in results/result_predictions_1000_tweets.csv

in the terminal : 
go to the repository of the folder
and then run the command :

streamlit run app.py

But if you want to train the model before running the web app

Run All cells  in SentimentPrediction.ipynb   
you will obtain result_predictions_1000_tweets.csv and  saved_steps1.pkl
_________________________________________________________________________________

You can also run twitter.py to get tweets  using Twitter API 
you will obtain result1.csv 

Anytime RUN :
streamlit run app.py
and upload any of the CSV files
or type your own statement to predict if it is positive or not
_________________________________________________________________________________

Contactez-moi : mohamedelkanfoudi2000@gmail.com

