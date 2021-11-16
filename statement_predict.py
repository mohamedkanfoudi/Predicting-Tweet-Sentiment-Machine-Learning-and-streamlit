import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re 
import string 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer





stop_words = set(stopwords.words('english'))


def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset
def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    #ps = PorterStemmer()
    #stemmed_words = [ps.stem(w) for w in filtered_words]
    #lemmatizer = WordNetLemmatizer()
    #lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)

def explore(df):
  # DATA
  st.write('Data:')
  st.write(df)
  # SUMMARY
  """
  df_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
  numerical_cols = df_types[~df_types['Data Type'].isin(['object',
                   'bool'])].index.values
  df_types['Count'] = df.count()
  df_types['Unique Values'] = df.nunique()
  df_types['Min'] = df[numerical_cols].min()
  df_types['Max'] = df[numerical_cols].max()
  df_types['Average'] = df[numerical_cols].mean()
  df_types['Median'] = df[numerical_cols].median()
  df_types['St. Dev.'] = df[numerical_cols].std()
  """
  st.write('Summary:')
  #st.write(df_types)


def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]

vectorTF = data["tf_vector"]


def convert_df(df):
        return df.to_csv().encode('utf-8')

def show_statement_predict():
    #st.title("Explore data ")

    st.write(
        """
    ### PREDICTION  |  Using : Logistics Regression model 
    """
    """
    # Accuracy score :  78.79%
    """
    )
    st.write("""### We need a statement to predict """)
    user_input = st.text_input("label goes here",value="")

    data = ['0','232999989','Thu Jun 25 07:10:05 PDT 2021','NO_QUERY','INMK69', user_input]
    df2 = pd.DataFrame(
                        [data],
                        columns =['predictions','tweet_id', 'created_at', 'query' , 'user', 'text' ])
    df2.to_csv('df_new.csv')
    df2 = pd.read_csv('df_new.csv')
    try : 
        df2 = remove_unwanted_cols(df2, ["Unnamed: 0","predictions", "created_at","query", "user"])
    except :     
            try : 
                df2 = remove_unwanted_cols(df2, ["predictions", "created_at","query", "user"])
            except : 
                try : 
                    df2 = remove_unwanted_cols(df2, [ "created_at","query", "user"])
                except : 
                    try : 
                        df2 = remove_unwanted_cols(df2, [ "query", "user"])
                    except : 
                        try : 
                            df2 = remove_unwanted_cols(df2, [ "user"])
                        except: 
                            df2 = pd.DataFrame(
                                data=df2.values,
                                columns =["tweet_id", "created_at", "text"])    
            else : 
                df2 = remove_unwanted_cols(df2, [])
    

    print(df2)
    explore(df2)
    """
    csv3 = convert_df(data)

    st.download_button(
            "Press to Download",
            csv3,
            "statement_prediction.csv",
            "text/csv",
            key='download-csv'
        )
    """

    
    

    # Convert the dictionary into DataFrame

    ok2 = st.button("Predict2")

    if ok2 :
    
        # Creating text feature
        df2.text = df2["text"].apply(preprocess_tweet_text)

        df2.text = df2["text"].apply(preprocess_tweet_text)
        test_feature = vectorTF.transform(np.array(df2.iloc[:, 1]).ravel())
        
        prediction = regressor.predict(test_feature)
        
        test_result_ds = pd.DataFrame({'tweet_id': df2.tweet_id,'text':df2.text ,'prediction':prediction})
        test_result = test_result_ds.groupby(['tweet_id']).max().reset_index()
        test_result.columns = ['tweet_id','text', 'predictions']
        test_result.predictions = test_result['predictions'].apply(int_to_string)
        explore(test_result)




      

