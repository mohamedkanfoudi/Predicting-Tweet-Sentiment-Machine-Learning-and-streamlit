import streamlit as st
import pandas as pd
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
  st.write(df_types)
def download_file(df, types, new_types, extension):
  for i, col in enumerate(df.columns):
    new_type = types[new_types[i]]
    if new_type:
      try:
        df[col] = df[col].astype(new_type)
      except:
        st.write('Could not convert', col, 'to', new_types[i])
def transform(df):
  frac = st.slider('Random sample (%)', 1, 100, 100)
  if frac < 100:
    df = df.sample(frac=frac/100)
  
  cols = st.multiselect('Columns'
                        ,df.columns.tolist()
                        ,df.columns.tolist())
  df = df[cols]
  types = {'-':None
           ,'Boolean': '?'
           ,'Byte': 'b'
           ,'Integer':'i'
           ,'Floating point': 'f' 
           ,'Date Time': 'M'
           ,'Time': 'm'
           ,'Unicode String':'U'
           ,'Object': 'O'}
  new_types = {}
  expander_types = st.beta_expander('Convert Data Types')
  for i, col in enumerate(df.columns):
    txt = 'Convert {} from {} to:'.format(col, df[col].dtypes)
    expander_types.markdown(txt, unsafe_allow_html=True)
    new_types[i] = expander_types.selectbox('Field to be converted:'
                                            ,[*types]
                                            ,index=0
                                            ,key=i)
  st.text(" \n") #break line
  # first col 15% the size of the second  
  col1, col2 = st.beta_columns([.15, 1])
  with col1:
    btn1 = st.button('Get CSV')
  with col2:
    btn2 = st.button('Get Pickle')
  if btn1:
    download_file(df, types, new_types, "csv")
  if btn2:
    download_file(df, types, new_types, "pickle")
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

def show_show_page():
  st.title('Show a dataset')
  st.write('A general purpose data exploration app')
  file = st.file_uploader("Upload file", type=['csv' 
                                             ,'xlsx'
                                             ,'pickle'])
  if not file:
    st.write("Upload a .csv or .xlsx file to get started")
    return
  df = get_df(file)

  explore(df)
show_show_page()