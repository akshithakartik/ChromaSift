<h1 align="center">ChromaSift 🧪</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="https://github.com/akshithakartik/ChromaSift#-description">Description</a></li>
    <li><a href="https://github.com/akshithakartik/ChromaSift#-demonstration">Demonstration</a></li>
    <li><a href="https://github.com/akshithakartik/ChromaSift#-pipeline">Pipeline</a></li>
    <li><a href="https://github.com/akshithakartik/ChromaSift#-features">Features</a></li>
    <li><a href="https://github.com/akshithakartik/ChromaSift#-installation">Installation</a></li>
    <li><a href="https://github.com/akshithakartik/ChromaSift#-neo4j-graph-database">Neo4j Graph Database</a></li>
  </ol>
</details>


## • Description

ChromaSift is a **specialized search engine** designed to filter and rank **chromatography-related documents** from a large corpus of scientific abstracts. Users can **query specific keywords** to retrieve documents, which are then ranked by relevance and assigned a **similarity score**. Additionally, ChromaSift allows users to access the full content of resulting documents by clicking on their titles, providing **direct links** to the corresponding scientific journals.


## • Demonstration

https://github.com/akshithakartik/ChromaSift/assets/112664522/9e6a2bf4-5eab-4599-8e31-f0a3ad5bb1a0

## • Pipeline

<p align="center">
  <img src="https://github.com/akshithakartik/ChromaSift/assets/112664522/60441492-86e6-4678-acf3-c1537d4b71a3" alt="ChromaSift Image">
</p>

1) **Data Collection and Pre-processing**: Abstracts from scientific journals like PubMed are scraped to create a corpus of scientific literature. The abstracts are pre-processed using techniques such as stopword removal and lemmatization to clean and standardize the text.

2) **Neo4j Graph Database**: The pre-processed abstracts are stored in a Neo4j database, enabling efficient querying and relationships analysis.

3) **ML Model**: *SciBERT*, a domain-specific adaptation of BERT tailored for scientific text, is used to generate embeddings for both documents and queries. *Cosine Similarity* and *BM-25* are combined to compute a hybrid score, which quantifies the percentage similarity between a document and the query.

4) **Streamlit**: The results are integrated into an interactive interface built with Streamlit, allowing users to input queries and view ranked documents based on their relevance.

## • Features

While existing platforms like PubMed, Google Scholar, Scopus, Web of Science, and SciFinder offer extensive databases and advanced search capabilities for accessing scientific literature, ChromaSift is specifically geared towards chromatography-related documents, providing a **focused and curated search experience** for users interested in this field.

<img src="https://github.com/akshithakartik/ChromaSift/assets/112664522/16085d81-cff0-4059-8cf5-aad58f0d0e95" align="right" height="350" width="420" />

ChromaSift provides **query flexibility** and **comprehensive results**. It goes beyond basic keyword matching by detecting and retrieving documents that are **indirectly related** to the entered keyword. 

For example, when the query "*amino acid*" is entered, a document with 22% similarity is returned. 
Interestingly, this document does not contain the phrase "*amino acid*" **anywhere** in its text. 

However, it is determined as relevant because it primarily discusses High-Performance Liquid Chromatography (HPLC) - the main method for amino acid analysis. 

This highlights ChromaSift's ability to identify documents that are indirectly related to the query by considering **broader context** and **connections within scientific literature**.

## • Installation

Install required dependencies with:

`pip install -r requirements.txt`

You may chose to use the abstracts provided, or scrape your own data.

## • Neo4j Graph Database

**Processes** and **Materials** associated with each pre-processed abstract were obtained using Open AI's GPT-3.5 Turbo. These were used as nodes for the Neo4j database with corresponding relationships with documents. From the scientific abstracts scraped, 7000 nodes and 11,300 relationships were formed. Here is a subset of the resulting Neo4j database:

https://github.com/akshithakartik/ChromaSift/assets/112664522/03d4adfe-943a-40bd-832f-40ba08207507

This database can be visualized and explored using the Neo4j dump file provided.




