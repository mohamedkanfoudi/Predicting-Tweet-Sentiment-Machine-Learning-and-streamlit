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

def explore(df):
  # DATA
  st.write('Data:')
  st.write(df)
  # SUMMARY
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
  st.write('Summary:')
  #st.write(df_types)

def get_df(file):
  # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(file)
  return df


def download_file(df, types, new_types, extension):
  for i, col in enumerate(df.columns):
    new_type = types[new_types[i]]
    if new_type:
      try:
        df[col] = df[col].astype(new_type)
      except:
        st.write('Could not convert', col, 'to', new_types[i])




def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]

vectorTF = data["tf_vector"]


def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset
def int_to_string(sentiment):
    if sentiment == 0:
        return "Negative"
    elif sentiment == 2:
        return "Neutral"
    else:
        return "Positive"

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
def get_feature_vector(train_fit):
            vector = TfidfVectorizer(sublinear_tf=True)
            vector.fit(train_fit)
            return vector


def show_predict_page():
    st.title("Sentiment  Prediction")

    st.write(
        """
    ### PREDICTION  |  Using : Logistics Regression model 
    """
    """
    # Accuracy score :  78.79%
    """
    )

    st.write("""### try executing the twitter streaming code  twitter.py """)
    st.write("""### meanwhile , stop running the code and upload result.csv  """)
    st.write("""###example :  result1.csv is already existed """)
    file = st.file_uploader("Upload file1", type=['csv' 
                                             ,'xlsx'
                                             ,'pickle'])
    if not file:
        st.write("Upload a .csv or .xlsx file to get started")
        return
    df = get_df(file)
    try : 
        df = remove_unwanted_cols(df, ["predictions", "created_at","query", "user"])
    except : 
        try : 
            df = remove_unwanted_cols(df, [ "created_at","query", "user"])
        except : 
            try : 
                df = remove_unwanted_cols(df, [ "query", "user"])
            except : 
                try : 
                    df = remove_unwanted_cols(df, [ "user"])
                except: 
                    df = pd.DataFrame(
                         data=df.values,
                         columns =["tweet_id", "created_at", "text"])    
    else : 
        df = remove_unwanted_cols(df, [])
    print(df)
    explore(df)

    

    
    ok = st.button("Predict")

    


    def convert_df(df):
        return df.to_csv().encode('utf-8')


    

        
    if ok:
        

        # Creating text feature
        df.text = df["text"].apply(preprocess_tweet_text)
        
        
        
        df.text = df["text"].apply(preprocess_tweet_text)
        test_feature = vectorTF.transform(np.array(df.iloc[:, 1]).ravel())
        
        prediction = regressor.predict(test_feature)
        
        test_result_ds = pd.DataFrame({'tweet_id': df.tweet_id,'text':df.text ,'prediction':prediction})
        test_result = test_result_ds.groupby(['tweet_id']).max().reset_index()
        test_result.columns = ['tweet_id','text', 'predictions']
        test_result.predictions = test_result['predictions'].apply(int_to_string)
        explore(test_result)
        

        csv = convert_df(test_result)

        st.download_button(
            "Press to Download",
            csv,
            "file_prediction_result.csv",
            "text/csv",
            key='download-csv'
        )

     
        

        
    