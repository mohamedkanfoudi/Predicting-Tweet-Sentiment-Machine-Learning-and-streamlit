# Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit
(I prefere using VS Code )
 
 Run the first cell   in  requirement.ipynb  
VS Code will demand to install Jupiter and python, install them  , then choose your interpreter of python , after that return to requirement.ipynb  and click Run ALL 
TO train and then Predict !

type in terminal  : 

python
>>>import nltk 
>>>nltk.download('all')

it takes time 


download
the CSV file  !
https://drive.google.com/file/d/1lcLdqyVG6mlJUom5q9BBxBdI3puRChTn/view?usp=sharing 

and save it in  Predicting-Tweet-Sentiment-Maching-Learning-and-streamlit FOLDER


Install Anaconda 
https://www.anaconda.com/products/individual#windows



![image](https://user-images.githubusercontent.com/76444482/142013402-a446b389-71e6-40a6-83cf-af6c6c448f85.png)
![image](https://user-images.githubusercontent.com/76444482/142013451-b4286abf-05f3-4fe8-9482-707d00190fed.png)

in the terminal appears , run:

pip install streamlit

______________________________________________________________
you can run directly web app  because the result of training is already saved in saved_steps1.pkl   using pickle  
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

_________________________________________________________________________________________________
le guide d'installation  d'Anaconda: https://docs.anaconda.com/anaconda/install/windows/
set up : https://docs.anaconda.com/anaconda/navigator/getting-started/#managing-environments

