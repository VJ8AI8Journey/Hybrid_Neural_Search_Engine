import os
# If you want GPU support, set CUDA_PATH and update PATH.
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
os.environ["PATH"] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

import re
import html
import csv
import json
from datasets import load_dataset
import spacy

# Attempt GPU usage via spaCy + CuPy
try:
    import cupy
    spacy.require_gpu()
    print("GPU is enabled for spaCy processing.")
except (ImportError, Exception) as e:
    print("GPU not available or CuPy not installed; running on CPU.")

# Load spaCy English model (disable parser and NER for speed)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


#######################
# BATCH Preprocessing
#######################
def batch_preprocess_texts(texts):
    """
    Process a list of texts with spaCy in batch mode using nlp.pipe().
    This approach is much faster than calling nlp() individually per text.
    Returns a list of processed strings.
    """
    cleaned_texts = []
    processed_input = []
    for text in texts:
        # Basic manual cleaning
        if text is None:
            text = ""
        text = html.unescape(text)
        text = text.lower()
        text = re.sub(r"<[^>]*>", " ", text)  # remove HTML tags
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
        processed_input.append(text)

    # spaCy in batch mode
    # Adjust batch_size if needed (try 512, 1024, 2048, etc.)
    for doc in nlp.pipe(processed_input, batch_size=512):
        tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct or token.is_space)]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts


def build_doc_dict_chunked(dataset, min_tokens=30, save_to_file=False, output_path="all_the_news_cleaned.csv",
                           chunk_size=5000):
    """
    Processes the dataset in chunks to avoid memory spikes and overhead.
    Each chunk is processed in batch mode with spaCy, significantly speeding up large-scale runs.
    Also prints progress at both the chunk level and every 1000 documents.
    """
    docs = []
    total = len(dataset)
    # We'll iterate in increments of 'chunk_size'
    for start_idx in range(0, total, chunk_size):
        end_idx = min(start_idx + chunk_size, total)
        # Slicing row-wise so that each record is a dict, not a string
        # This returns a smaller dataset slice
        batch = dataset.select(range(start_idx, end_idx))

        # Extract columns as lists
        titles = batch["title"]
        articles = batch["article"]
        publications = batch["publication"]
        dates = batch["date"]
        authors = batch["author"]

        # Preprocess in batches
        title_clean_list = batch_preprocess_texts(titles)
        article_clean_list = batch_preprocess_texts(articles)

        for i in range(len(batch)):
            # Fine-grained progress print
            if i % 1000 == 0:
                print(f"  -> Processing doc {start_idx + i} / {total}")

            publication = publications[i] or ''
            date = dates[i] or ''
            author = authors[i] or ''

            combined_text = title_clean_list[i] + ". " + article_clean_list[i]
            if len(combined_text.split()) < min_tokens:
                continue

            doc = {
                "id": str(start_idx + i),
                "contents": combined_text,
                "publication": publication,
                "date": date,
                "author": author
            }
            docs.append(doc)

        print(f"Processed records {start_idx} - {end_idx} / {total}")

    if save_to_file:
        with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
            fieldnames = ['id', 'contents', 'publication', 'date', 'author']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for doc in docs:
                writer.writerow(doc)
        print(f"Documents saved to {output_path}")

    return docs


###########################################
# Utility functions
###########################################
def get_token_count(text: str) -> int:
    if text is None:
        return 0
    return len(text.split())


def print_doc_summary(docs, field='contents'):
    total_records = len(docs)
    total_tokens = sum(get_token_count(doc.get(field, "")) for doc in docs)
    avg_tokens = total_tokens / total_records if total_records > 0 else 0
    print("Cleaned Document Summary (after preprocessing):")
    print(f"  Total records: {total_records}")
    print(f"  Average token count in '{field}': {avg_tokens:.2f}")


def print_sample_docs(docs, num_samples=3):
    print("\nSample records from the cleaned document dictionary:")
    for doc in docs[:num_samples]:
        print(doc)
        print("-" * 80)


def print_raw_dataset_summary(dataset, field='article'):
    total_records = len(dataset)
    total_tokens = 0
    for record in dataset:
        text = record.get(field) or ''
        total_tokens += get_token_count(text)
    avg_tokens = total_tokens / total_records if total_records > 0 else 0
    print("Raw Dataset Summary (before preprocessing):")
    print(f"  Total records: {total_records}")
    print(f"  Average token count in '{field}': {avg_tokens:.2f}")
    return total_records, avg_tokens


###########################################
# Main
###########################################
def main():
    # We'll process 25% to keep runtime feasible
    dataset = load_dataset("rjac/all-the-news-2-1-Component-one", split="train[:25%]")
    print("Dataset loaded successfully!")
    print(f"Number of examples in dataset: {len(dataset)}")

    # Print summary
    print_raw_dataset_summary(dataset, field='article')

    # Build doc dictionary in chunks
    docs = build_doc_dict_chunked(
        dataset,
        min_tokens=30,
        save_to_file=True,
        output_path="all_the_news_cleaned.csv",
        chunk_size=5000  # Adjust chunk size based on memory/hardware
    )
    print("\nPreprocessing complete! Documents are stored in memory as a list of dictionaries.\n")

    print_doc_summary(docs, field='contents')
    print_sample_docs(docs, num_samples=3)


if __name__ == "__main__":
    main()
