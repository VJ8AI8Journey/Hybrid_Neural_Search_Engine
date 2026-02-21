# Hybrid Search Engine Project

This project implements a hybrid search engine that combines BM25 lexical retrieval with SBERT neural reranking. It includes preprocessing, a command-line interface for search, and an evaluation framework.

## Project Components

### Files Included

* **preprocessing.py**: Cleans and processes the raw dataset. It handles HTML tags removal, tokenization, lemmatization, and stopword removal using spaCy. The processed data is saved to a CSV file.

* **bm25_merged_SBERT_cli_final.py**: Implements the search engine functionality with:
  - BM25 algorithm for initial document retrieval
  - SBERT cross-encoder for relevance reranking
  - A spell-checking function to correct user queries
  - Interactive command-line interface for searching and viewing results

* **BM25_SBERT_Evaluation.ipynb**: Evaluates the search engine performance by:
  - Implementing both BM25 and BM25+SBERT models
  - Running experiments with predefined queries
  - Calculating evaluation metrics (Precision@k, nDCG@k, MAP, F1)
  - Generating visualization plots comparing model performance
  - The csv files generated as part of evaluations are in a evaluation_results folder

* **all_the_news_cleaned.csv**: The preprocessed dataset containing cleaned news articles ready for search, which is 25% of the actual All The News 2.0 dataset. Contains 660,000 rows. This is not being submitted as file size is over 1GB in size. This is the data that is used in the actual code, and while demonstrating the search engine.

* **all_the_news_cleaned_TEST.csv**: This is the file that is being added to submission due to file size constraints - only contains 1000 rows.

To run the code smoothly, the csv_path must be changed to reflect this. However, the TEST csv file doesn't contain full data, hampering performance and usability.

## Workflow

1. **Data Preprocessing**:
   - Run `preprocessing.py` first to clean the raw dataset
   - This creates `all_the_news_cleaned.csv` with preprocessed documents

2. **Using the Search Engine**:
   - Run `bm25_merged_SBERT_cli_final.py` to start the interactive search interface
   - Enter search queries and view ranked results

3. **Evaluation**:
   - Run `BM25_SBERT_Evaluation.ipynb` to evaluate search performance
   - This generates performance metrics and comparison visualizations
   - Results are saved as tables and plots which are displayed in the notebook.
   - The results of the Search Engine are saved as csv files in the evaluation_results folder.

## Requirements

The project requires Python libraries including:
- rank_bm25
- sentence_transformers
- spacy
- textblob
- pandas
- matplotlib
- seaborn
- numpy

GPU acceleration is supported for Apple Silicon (via MPS) and NVIDIA GPUs (via CUDA) when available.
