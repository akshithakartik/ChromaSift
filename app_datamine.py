from my_module import generate_Chrom_docs, generate_relevant_docs, create_df, preprocess, create_positional_index, doc_matches, show_results, proximity_query, show_proxy_results, bert_preprocess

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time as time
import math
import re

col1, col2, col3 = st.columns([1, 10, 1]) 

with col2:
    st.image('finalheading.png', width=500)


css = """
.st-emotion-cache-7fawd5 {
    position: absolute;
    background: rgba(0, 0, 0, 0);
    color: rgb(0, 0, 0);
    inset: 0px;
    color-scheme: revert;
    overflow: hidden;
}
"""

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("**Welcome to ChromaSift, a searchable database designed for you at Merck to seamlessly retrieve and access academic papers across chromatographic conditions.**  \n\n**Type in any query to generate a series of titles ranked by their relevance, indicated by a percentage in similarity to the right of each result.**")


file_contents, labels, link_contents = create_df()

df = pd.DataFrame({
    "Text": file_contents,
    "Label": labels,
    "Link": link_contents,
})

df_display = pd.DataFrame({
    "Text": file_contents,
})


df['Text'] = df['Text'].apply(bert_preprocess)
documents = df.Text
preprocessed_documents = [preprocess(doc) for doc in documents]

file_path_pickle = "C:\\Users\\Akshitha\\Downloads\\data2.pickle"

with open(file_path_pickle, 'rb') as file:
    final_embeddings = pickle.load(file)

final_embeddings = np.array(final_embeddings)

df_positive = generate_Chrom_docs(preprocessed_documents, df, final_embeddings)
#st.dataframe(df_positive['Text'], width=1000, height=425)

st.header('Filter Chromatography Documents by Query')

labels = len(df)

#file_path_pickle = "C:\\Users\\Akshitha\\Downloads\\dictionary.pickle"

if "counter" not in st.session_state:
    st.session_state.counter = 0

if "keyword_list" not in st.session_state:
    st.session_state.keyword_list = []

# with open(file_path_pickle, 'rb') as file:
#     dictionary = pickle.load(file)

# for doc in range(labels):
#   words = preprocessed_documents[doc]
#   #create_positional_index(words, doc, dictionary)


search = st.text_input('Enter query:')

if search:

  if (len(search) != 0):
    
    if (search not in st.session_state.keyword_list):
      st.session_state.keyword_list.append(search)
      st.session_state.counter = 0
    else:
      st.session_state.counter += 1

    if (st.session_state.counter == 0):
      progress_text = "Retrieving documents..."
      my_bar = st.progress(0, text=progress_text)

      for percent_complete in range(100):
          time.sleep(0.1)
          my_bar.progress(percent_complete + 1, text=progress_text)
      
      my_bar.text("")

    start = time.time()
    query = search

    df_query = generate_relevant_docs(query, preprocessed_documents, df, final_embeddings, labels)

    doc = doc_matches(labels, df, query)

    doc_set = set(doc)

    df_final = df_query[df_query.index.isin(doc_set)]
    ranking_1 = df_final.sort_values(by='Hybrid_Test_Scores_Query', ascending=False)

    df_final = df_query[~df_query.index.isin(doc_set)]
    ranking_2 = df_final.sort_values(by='Hybrid_Test_Scores_Query', ascending=False)

    result = pd.concat([ranking_1, ranking_2], axis=0)
    result_count = len(result)

    if (result_count == 0):
      st.error('No documents match your search!', icon="🚨")
    else:
      time_display = time.time() - start
  
      #st.write(f"*{result_count} results in {time_display:.2f}s*")

      results_per_page = 10
      pages = math.ceil(len(result) / results_per_page)
      l = list(range(1, pages + 1))

      with st.sidebar:
        option = st.selectbox(
            'Choose the page you would like to navigate to:', l)
      
      st.write(f"*{result_count} results in {time_display:.2f}s*")
          
      runs = 1
      
      with st.spinner('Loading page...'):
        time.sleep(5)
        

      start_idx = (option - 1) * results_per_page
      end_idx = min(start_idx + results_per_page, len(result))

      st.write(f"*Showing results {start_idx + 1}-{end_idx}*")

      show_results(query, result, start_idx, end_idx)

  else:
    st.warning('Enter a valid keyword!', icon="⚠️")

  

# st.header('Filter Chromatography Documents by Proximity Search')

# submitted = ""

# with st.form("user_form"):
#    st.header("Proximity Search")

#    query1 = st.text_input('Enter the first query:')
#    query2 = st.text_input('Enter the second query:')
#    skip = st.slider("How many words apart?")

#    submitted = st.form_submit_button("Submit")

# if submitted:
#   progress_text = "Retrieving documents..."
#   my_bar = st.progress(0, text=progress_text)

#   for percent_complete in range(100):
#       time.sleep(0.1)
#       my_bar.progress(percent_complete + 1, text=progress_text)
  
#   my_bar.text("")
#   matching_docs = proximity_query(preprocess(query1), preprocess(query2), skip, dictionary)
#   show_proxy_results(query1, query2, df, matching_docs)


