import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle

import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification

from sklearn.metrics.pairwise import cosine_similarity

import nltk
import re
import html
import langchain
import streamlit as st
from st_circular_progress import CircularProgress

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from rank_bm25 import BM25Okapi

nltk.download('punkt')
nltk.download('stopwords')

pretrained_model = 'allenai/scibert_scivocab_uncased'

sciBERT_tokenizer = BertTokenizer.from_pretrained(pretrained_model,
                                          do_lower_case=True)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model,
                                                          output_attentions=False,
                                                          output_hidden_states=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 40,
    length_function = len,
    is_separator_regex = False,
)


# Replaces common abbreviations with their word expansions
def subs_abb(text):
  text = re.sub(r"\bLC\b", "Liquid Chromatography", text)
  text = re.sub(r"\bGC\b", "Gas Chromatography", text)
  text = re.sub(r"\bHPLC\b", "High Performance Liquid Chromatography", text)
  text = re.sub(r"\bMS\b", "Mass Spectrometry", text)

  return text

# Performs text preprocessing: tokenization, lemmatization, stopword removal
def preprocess(text):

    punctuation = [';', ':', '-', '–', '/', ',']

    for i in punctuation:
      text = text.replace(i, ' ')

    text = subs_abb(text)

    text = text.lower()

    words = word_tokenize(text)

    stemmer = PorterStemmer()

    stop_words = set(stopwords.words('english'))

    words = [stemmer.stem(word) for word in words if word not in stop_words]

    return words

# Removes HTML sequences
def bert_preprocess(text_with_entities):

  decoded_text = html.unescape(text_with_entities)

  clean_text = decoded_text.replace('<i>', '').replace('</i>', '')
  clean_text = clean_text.replace('<sub>', '').replace('</sub>', '')
  clean_text = clean_text.replace('<sup>', '').replace('</sup>', '')
  clean_text = clean_text.replace('<h4>', '').replace('</h4>', '')
  clean_text = clean_text.replace('<b>', '').replace('</b>', '')

  pattern = re.compile(r'\\x[0-9a-fA-F]{2}')

  clean_text = pattern.sub('', clean_text)

  clean_text = clean_text.replace("\xa0", "")
  clean_text = clean_text.replace("\\", "")

  return clean_text



# Creates a dataframe containing documents from the Drive
def create_df():
  data_directory = "C:\\Users\\Akshitha\\Downloads\\ChromatographyData"
  
  link_directory = "C:\\Users\\Akshitha\\Downloads\\links"

  file_paths = []
  labels = []
  link_contents = []

  for label in ["Non-related", "Related"]:
    directory_path = os.path.join(data_directory, label)
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_paths.append(file_path)
        labels.append(label)

        if label == "Related":
          if filename.startswith("PMC"):
            base_name, extension = os.path.splitext(filename)
            link_content = "https://www.ncbi.nlm.nih.gov/pmc/articles/" + base_name + "/"
          else:
            link_path = os.path.join(link_directory, filename)
            with open(link_path, "r") as link1:
              link_temp = link1.read()
              link_content = link_temp[len("Link: "):].strip()
        else:
          link_content = "null"
        link_contents.append(link_content)

  file_contents = []

  for file_path in file_paths:
    with open(file_path, "r", encoding='utf-8') as fil1:
        file_content = fil1.read()
        file_contents.append(file_content)
  
  return file_contents, labels, link_contents

# Converts input text into its vector embedding
def convert_single_abstract_to_embedding(tokenizer, model, in_text, MAX_LEN):

    input_ids = tokenizer.encode(
                        in_text,
                        add_special_tokens = True,
                        max_length = MAX_LEN,
                   )

    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long",
                              truncating="post", padding="post")

    input_ids = results[0]

    attention_mask = [int(i>0) for i in input_ids]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        logits, encoded_layers = model(
                                    input_ids = input_ids,
                                    token_type_ids = None,
                                    attention_mask = attention_mask,
                                    return_dict=False)

    layer_i = 12
    batch_i = 0
    token_i = 0

    embedding = encoded_layers[layer_i][batch_i][token_i]

    embedding = embedding.detach().cpu().numpy()

    return(embedding)


# Calculates cosine similarity scores between a query and document embeddings
def calculate_cosine_similarity_scores(query_text, final_embeddings):

    query_vect = convert_single_abstract_to_embedding(sciBERT_tokenizer, model, query_text, 512)

    query_vect = np.array(query_vect)
    query_vect = query_vect.reshape(1, -1)

    query_vect = query_vect.squeeze()

    cos_similarities = [cosine_similarity([query_vect], [embedding])[0][0] for embedding in final_embeddings]

    return(cos_similarities)


# Normalizes scores to the range 0-1 
def normalize_scaling(scores):
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score == 0:
        return scores

    if min_score == max_score:
        return [1.0] * len(scores)

    scaled_scores = [(score - min_score) / (max_score - min_score) for score in scores]

    return scaled_scores


# Returns documents where the 2 input queries separated by a certain radius
def proximity_query(query1, query2, skip, pos_dict):

  word1 = pos_dict.get(query1[0])
  word2 = pos_dict.get(query2[0])

  common_docs = set(word1).intersection(word2)

  matching_docs = []

  for doc in common_docs:

    occ1 = word1[doc]
    occ2 = word2[doc]

    len1 = len(occ1)
    len2 = len(occ2)

    i = j = 0

    while (i != len1) & (j != len2):

        diff = abs(occ2[j] - occ1[i])

        if (diff <= skip):
          matching_docs.append(doc)
          break
        elif (occ2[j] > occ1[i]):
          i+=1
        else:
          j+=1

  return matching_docs


def phrase_query(preprocessed_query, document_set, dictionary):

  length = len(preprocessed_query)

  for x in range(1, length):
    document_set = proximity_query(preprocessed_query, x, dictionary, document_set)

  return document_set

 
def create_positional_index(words, doc_index, dictionary):

  temp_dict = {}

  pos = 0

  for word in words:
    key = word
    temp_dict.setdefault(key, [])
    temp_dict[key].append(pos)
    pos += 1

  for x in temp_dict:
    if dictionary.get(x):
      dictionary[x][doc_index] = temp_dict.get(x)
    else:
      key = x
      dictionary.setdefault(key, [])
      dictionary[key] = {}
      dictionary[key][doc_index] = temp_dict.get(x)


# Returns parts of the input text that match the phrase
def exact_phrase_matching(text, phrase):
    escaped_phrase = re.escape(phrase)
    
    pattern = re.compile(r'\b' + escaped_phrase + r'\b', re.IGNORECASE)

    matches = pattern.findall(text)

    return matches

# Returns documents that exactly contain the phrase entered
def doc_matches(labels, df, keyword):
  
  doc = []
  
  for i in range(labels):
    abstract = df.iloc[i].Text
    abstract = subs_abb(abstract)
    matches = exact_phrase_matching(abstract, keyword)
    
    if (len(matches) > 0):
      doc.append(i)
  
  return doc


def extract_keyword_phrases(abstract, keyword, window_size):
    #pattern = re.compile(r'\b(?:\S+\s+){0,' + str(window_size) + r'}' + re.escape(keyword) + r'(?:\s+\S+){0,' + str(window_size) + r'}\b', re.IGNORECASE)

    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)

    keyword_phrases = pattern.findall(abstract)

    return keyword_phrases


# Returns documents relevant to Chromatography
def generate_Chrom_docs(preprocessed_documents, df, embeddings):

    query_text = "Chromatography"

    cosine_similarities = calculate_cosine_similarity_scores(query_text, embeddings)
    df['Cosine_Similarity'] = cosine_similarities

    preprocessed_query = preprocess(query_text)

    bm25 = BM25Okapi(preprocessed_documents)

    bm25_scores = bm25.get_scores(preprocessed_query)

    normalized_bm25 = normalize_scaling(bm25_scores)
    df['BM_25'] = normalized_bm25

    cosine_similarities = np.array(cosine_similarities)
    normalized_bm25 = np.array(normalized_bm25)

    hybrid_test_scores = (0.8 * normalized_bm25) + (0.2 * cosine_similarities)
    df['Hybrid_Test_Scores'] = hybrid_test_scores

    df_positive = df[df['Hybrid_Test_Scores'] > 0.15]
    ranking = df_positive.sort_values(by='Hybrid_Test_Scores', ascending=False)

    return ranking


# Returns documents relevant to a specific keyword
@st.cache_data
def generate_relevant_docs(keyword, preprocessed_documents, df, embeddings, labels):

    query_text = keyword
    cosine_similarities = calculate_cosine_similarity_scores(query_text, embeddings)
    df['Cosine_Similarity_Query'] = cosine_similarities

    preprocessed_query = preprocess(query_text)

    bm25 = BM25Okapi(preprocessed_documents)

    bm25_scores = bm25.get_scores(preprocessed_query)

    normalized_bm25 = normalize_scaling(bm25_scores)
    df['BM_25_Query'] = normalized_bm25

    cosine_similarities = np.array(cosine_similarities)
    normalized_bm25 = np.array(normalized_bm25)

    hybrid_test_scores_query = (0.8 * normalized_bm25) + (0.2 * cosine_similarities)
    df['Hybrid_Test_Scores_Query'] = hybrid_test_scores_query

    #document_set = set(range(labels))

    # if (len(preprocessed_query) > 1):
    #   document_set = phrase_query(preprocessed_query, document_set, dictionary)

    df_positive_query = df[(df['Hybrid_Test_Scores'] > 0.15) & (df['Hybrid_Test_Scores_Query'] > 0.15)]

    return df_positive_query


def get_horizontal_bar(score):
    # Define colors for the gradient (green to yellow to red)
    colors = ['#008000', '#FFFF00', '#FF0000']
    
    # Calculate color based on the score
    if score >= 0.7:
        color = colors[0]
    elif score >= 0.5:
        color = colors[1]
    else:
        color = colors[2]
    
    # Calculate width of the bar based on the score
    width = int(score * 100)
    
    # Return HTML code for the horizontal bar
    return f"""
    <div style='background: {color}; width: {width}%; height: 5px;'></div>
    """

# Displays the document ranking results in Streamlit
def show_results(query_text, df_ranking, start, end):

  for doc in range(start, end):

    input_text = df_ranking.iloc[doc].Text
    custom_texts = text_splitter.create_documents([input_text])
    custom_texts = [text.page_content for text in custom_texts]

    title = subs_abb(custom_texts[0])
    title = (title[:65] + '..') if len(title) > 65 else title
    title = title[7:]

    score = df_ranking.iloc[doc].Hybrid_Test_Scores_Query
    score = round(100 * score)

    #print(score)
    #print(st.session_state.counter)

    #pattern = re.compile(re.escape(query_text), re.IGNORECASE)
    pattern = re.compile(r'\b' + re.escape(query_text) + r'\b', re.IGNORECASE)
    
    result = pattern.sub(f"**{query_text.upper()}**", input_text)

    fixed_text = result.replace('\n\n', '').replace('\n', '').strip()
    fixed_text = re.sub(r'\s+', ' ', fixed_text.strip())

    title_strip, abstract = fixed_text.split("Abstract:", 1)
    formatted_text = f"{title_strip}  \n\nAbstract: {abstract}"

    link = df_ranking.iloc[doc].Link

    #st.subheader(f"{doc + 1}. {title}")

    link_style = "color: black; text-decoration: none;"

    st.markdown(f"<h3><a href='{link}' style='{link_style}'>{doc + 1}. {title}</a></h3>", unsafe_allow_html=True)

    # make columns for display
    cols = st.columns([4, 1])
    
    # left column
    with cols[0]:
        st.markdown(formatted_text)
      
    # right column
    with cols[1]:
      circular_percentage = CircularProgress(
                label="Similarity Score",
                value=score,
                key=doc,
                color="teal"
            )

      circular_percentage.update_value(progress=score)
      circular_percentage.st_circular_progress()


    #plot_spot.empty()   
    st.write("---")


def show_proxy_results(query1, query2, df_ranking, matching_docs_index):

  for doc in matching_docs_index:

    input_text = df_ranking.iloc[doc].Text
    custom_texts = text_splitter.create_documents([input_text])
    custom_texts = [text.page_content for text in custom_texts]

    title = subs_abb(custom_texts[0])
    title = (title[:65] + '..') if len(title) > 65 else title
    title = title[7:]

    #print(title)
    st.subheader(f"{doc + 1}. {title}")

    if (len(custom_texts) > 1):

      custom_texts[1] = custom_texts[1][10:]

      custom_texts = list(custom_texts)

      text = "".join(custom_texts)
      text = subs_abb(text)

      word1 = query1
      word2 = query2

      fin_res = []

      phrases1 = extract_keyword_phrases(text, query1, 5)
      phrases2 = extract_keyword_phrases(text, query2, 5)

      phrases = phrases1 + phrases2

      for phrase in phrases:

        replacement1 = "**" + word1.upper() + "**"
        replacement2 = "**" + word2.upper() + "**"

        pattern1 = re.compile(re.escape(word1), re.IGNORECASE)
        pattern2 = re.compile(re.escape(word2), re.IGNORECASE)

        resulting = pattern1.sub(replacement1, phrase)
        resulting = pattern2.sub(replacement2, phrase)

        fin_res.append(resulting)
      
      fin_res = list(fin_res)
      text = "...".join(fin_res)
      if (len(fin_res) > 0):
        text += "..."
      else :
        text += "No explicit query matches"
      st.markdown(text)
    else:
      st.write('No abstract available')
    
    if st.button("View Document", key=doc, type="primary"):
      st.write(input_text)

