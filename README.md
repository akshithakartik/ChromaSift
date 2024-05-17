<h1 align="center">ChromaSift 🧪</h1>


## • Description

ChromaSift is a **specialized search engine** designed to filter and rank **chromatography-related documents** from a large corpus of scientific abstracts. Users can **query specific keywords** to retrieve documents, which are then ranked by relevance and assigned a **similarity score**. Additionally, ChromaSift allows users to access the full content of resulting documents by clicking on their titles, providing direct links to the corresponding scientific journals.


## • Demonstration

https://github.com/akshithakartik/ChromaSift/assets/112664522/9e6a2bf4-5eab-4599-8e31-f0a3ad5bb1a0

## • Pipeline

![image](https://github.com/akshithakartik/ChromaSift/assets/112664522/60441492-86e6-4678-acf3-c1537d4b71a3)

Scientific journals like PubMed are first scraped to obtain a corpus of abstracts. These are then pre-processed, using methods like stopword removal and lemmatization, and inputted to the Neo4j graph database and ML model. The ML phase of the pipeline consists of SciBERT - a domain-specific adaptation of BERT tailored specifically for scientific text - which is utilized to obtain document and query embeddings. Metrics like Cosine Similarity and BM-25 are combined to form a hybrid score, reflecting the percentage similarity of a document to the query. Finally, this is integrated with Streamlit to create an interactive interface for users.


