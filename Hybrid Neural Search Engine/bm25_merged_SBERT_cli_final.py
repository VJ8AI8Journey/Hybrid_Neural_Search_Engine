import csv
from rank_bm25 import BM25Okapi
import string
import re
import html
import json
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import CrossEncoder
import numpy as np
import csv
from textblob import TextBlob
from collections import Counter

csv.field_size_limit(1000000)


class SimpleSpellChecker:
    """
    Simplified spell checker that focuses on efficiency while maintaining effectiveness.
    Uses a limited set of important terms and simplified correction logic.
    """

    def __init__(self, docs=None, max_terms=5000, min_term_freq=5):
        """
        Initialize the simplified spell checker.

        Args:
            docs: List of documents to extract important terms from
            max_terms: Maximum number of important terms to store
            min_term_freq: Minimum frequency for a term to be considered important
        """
        self.important_terms = set()
        self.term_frequencies = Counter()
        self.max_terms = max_terms
        self.min_term_freq = min_term_freq
        self.correction_cache = {}  # Cache for previously seen corrections

        # Add common news terms that should always be preserved
        self.common_news_terms = {
            'france', 'germany', 'uk', 'usa', 'china', 'japan', 'russia', 
            'ukraine', 'brexit', 'covid', 'trump', 'biden', 'economy',
            'climate', 'change', 'vaccine', 'campaign', 'election', 'administration',
            'president', 'minister', 'government', 'parliament', 'congress',
            'senate', 'house', 'court', 'justice', 'law', 'policy', 'politician',
            'democrat', 'republican', 'liberal', 'conservative', 'progressive',
            'pandemic', 'virus', 'disease', 'health', 'medicine', 'doctor', 'hospital',
            'school', 'university', 'college', 'education', 'student', 'teacher',
            'business', 'company', 'corporation', 'industry', 'market', 'stock',
            'technology', 'computer', 'internet', 'software', 'hardware', 'app',
            'environment', 'pollution', 'warming', 'renewable', 'sustainable',
            'military', 'army', 'navy', 'air force', 'defense', 'weapon', 'war',
            'peace', 'treaty', 'agreement', 'negotiation', 'diplomacy', 'embassy'
        }
        self.important_terms.update(self.common_news_terms)

        # Common misspellings dictionary
        self.common_misspellings = {
            'frnace': 'france',
            'ukrane': 'ukraine',
            'brxit': 'brexit',
            'vacine': 'vaccine',
            'econmy': 'economy',
            'campain': 'campaign',
            'adminstration': 'administration',
            'chagne': 'change',
            'microsft': 'microsoft',
            'newyork': 'new york',
            'trumps': 'trump',
            'bidens': 'biden',
            'govrnment': 'government',
            'parlment': 'parliament',
            'politcs': 'politics'
        }

        # Learn important terms if docs are provided
        if docs:
            self.learn_important_terms(docs)
            print(f"Learned {len(self.important_terms)} important terms")

    def learn_important_terms(self, docs):
        """
        Learn important terms from the entire corpus using a streaming approach.
        Processes documents in batches to maintain memory efficiency.

        Args:
            docs: List of documents to learn from
        """
        print("Learning important terms from corpus (optimized for performance)...")
        
        # Process documents in batches to avoid memory issues
        batch_size = 1000
        total_docs = len(docs)
        
        # Use a temporary counter for each batch
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = docs[batch_start:batch_end]
            
            # Process each document in the batch
            for doc in batch:
                if isinstance(doc, dict) and 'contents' in doc:
                    # Only process the first 500 characters of each document for efficiency
                    text = doc['contents'][:500].lower()
                    # Extract words, removing punctuation
                    words = re.findall(r'\b[a-z]+\b', text)
                    # Update term frequencies
                    self.term_frequencies.update(words)
            
            # Print progress
            print(f"Processed {batch_end}/{total_docs} documents")
        
        # Select top terms by frequency
        for term, freq in self.term_frequencies.most_common(self.max_terms):
            if freq >= self.min_term_freq and len(term) > 3:  # Only consider terms with reasonable length
                self.important_terms.add(term)

    def is_important_term(self, term):
        """
        Check if a term is considered important.

        Args:
            term: Term to check

        Returns:
            Boolean indicating if term is important
        """
        return term.lower() in self.important_terms

    def find_closest_important_term(self, word):
        """
        Find the closest important term to a given word.
        Uses a simplified approach focusing on prefix matching.

        Args:
            word: Word to find closest important term for

        Returns:
            Closest important term or None if no close match
        """
        # Check common misspellings first
        if word.lower() in self.common_misspellings:
            return self.common_misspellings[word.lower()]

        if not self.important_terms or len(word) < 3:
            return None

        # Simple prefix matching for efficiency
        prefix = word[:3].lower()
        
        # Early termination - only check a limited number of candidates
        candidates = []
        count = 0
        for term in self.important_terms:
            if term.startswith(prefix):
                candidates.append(term)
                count += 1
                if count >= 50:  # Limit number of candidates for performance
                    break

        if not candidates:
            return None

        # Find closest match by simple character comparison
        min_distance = float('inf')
        closest_term = None

        for term in candidates:
            # Only consider terms with similar length
            if abs(len(term) - len(word)) > 2:
                continue

            # Simple character-by-character comparison with early termination
            distance = 0
            for a, b in zip(word.lower(), term):
                if a != b:
                    distance += 1
                    if distance > 2:  # Early termination if distance exceeds threshold
                        break
            
            # Add penalty for length difference
            distance += abs(len(word) - len(term))

            if distance < min_distance and distance <= 2:  # Only consider close matches
                min_distance = distance
                closest_term = term

        return closest_term

    def correct_spelling(self, query):
        """
        Correct spelling in a query with a focus on efficiency.

        Args:
            query: User input query string

        Returns:
            Tuple of (corrected_query, was_corrected, correction_details)
        """
        # Check cache first for performance
        if query in self.correction_cache:
            return self.correction_cache[query]
            
        # Split the query into words
        words = query.split()
        corrected_words = []
        corrections = {}
        was_corrected = False

        for word in words:
            # Skip short words
            if len(word) < 3:
                corrected_words.append(word)
                continue

            # Handle apostrophes (e.g., trump's -> trump)
            word_clean = word.lower().replace("'s", "")
            
            # Check if this is an important term or close to one
            closest_term = self.find_closest_important_term(word_clean)

            if closest_term and closest_term.lower() != word_clean.lower():
                # Found a close match to an important term
                was_corrected = True
                corrections[word] = closest_term
                corrected_words.append(closest_term)
            elif self.is_important_term(word_clean):
                # Word is already an important term
                corrected_words.append(word)
            else:
                # Use TextBlob for general vocabulary, but only for words that might need correction
                # Simplified heuristic to reduce TextBlob usage
                if len(word) > 4 and re.search(r'[aeiou]{3}|[^aeiou]{4}', word_clean):
                    corrected_word = str(TextBlob(word_clean).correct())
                    if corrected_word.lower() != word_clean.lower():
                        was_corrected = True
                        corrections[word] = corrected_word
                        corrected_words.append(corrected_word)
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)

        # Join the corrected words back into a query
        corrected_query = ' '.join(corrected_words)
        
        # Cache the result
        result = (corrected_query, was_corrected, corrections)
        self.correction_cache[query] = result
        
        return result


def load_docs_from_csv(csv_path="all_the_news_cleaned.csv"):
    """
    Load preprocessed documents from a CSV file.
    Each row is converted into a dictionary.
    """
    docs = []
    with open(csv_path, "r", newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            docs.append(row)
    return docs


def simple_tokenizer(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text.split()


def bm25_algorithm(docs, query, top_k=10):
    query_tokenized = query.lower().split()
    tokenized = []
    for doc in docs:
        combined = f"{doc['contents']} {doc['author']} {doc['publication']} {doc['date']}"
        tokenized.append(simple_tokenizer(combined))
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query_tokenized)
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    return [docs[i] for i in ranked_indices]


def sbert_rerank(bm25_top_k_docs, query, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', top_k=5):
    """
    Re-rank BM25 results using SBERT cross-encoder for better relevance.

    Args:
        bm25_top_k_docs: Top documents from BM25 ranking
        query: Original search query
        model_name: SBERT cross-encoder model to use
        top_k: Number of documents to return after re-ranking

    Returns:
        List of re-ranked documents with their SBERT scores
    """
    # Initialize the cross-encoder model
    model = CrossEncoder(model_name)

    # Prepare query-document pairs for scoring
    pairs = [(query, doc['contents']) for doc in bm25_top_k_docs]

    # Get scores from SBERT
    sbert_scores = model.predict(pairs)

    # Combine documents with their scores
    scored_docs = list(zip(bm25_top_k_docs, sbert_scores))

    # Sort by SBERT score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top k documents with their scores
    return [{"doc": doc, "score": float(score)} for doc, score in scored_docs[:top_k]]


def print_results(results, show_full_content=False):
    """
    Print search results in a readable format.
    
    Args:
        results: List of result documents with scores
        show_full_content: Whether to show full content or just a snippet
    """
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        doc = result['doc']
        score = result['score']
        
        print(f"\nRank {i} (Score: {score:.4f})")
        print(f"Publication: {doc['publication']}")
        print(f"Date: {doc['date']}")
        print(f"Author: {doc['author']}")
        
        if show_full_content:
            print(f"Content: {doc['contents']}")
        else:
            snippet = doc["contents"][:200] + "..." if len(doc["contents"]) > 200 else doc["contents"]
            print(f"Content snippet: {snippet}")
            print("(Type 'view N' to see full content of result N)")


def main():
    # Load data and build indices
    print("Loading documents and building indices...")
    csv_path = "all_the_news_cleaned.csv"

    try:
        docs = load_docs_from_csv(csv_path)
        print(f"Loaded {len(docs)} documents from {csv_path}")

        # Initialize simplified spell checker with the loaded documents
        print("Learning important terms from the entire corpus...")
        spell_checker = SimpleSpellChecker(docs)

    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_path}'")
        print("Please ensure the CSV file is in the correct location and try again.")
        return
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return

    # CLI loop
    print("\n===== BM25 + SBERT Document Retrieval System =====")
    print("Type your search query or use one of these commands:")
    print("  'exit' - Quit the program")
    print("  'view N' - View full content of result N")
    print("  'bm25' - Show BM25 results for the last query")
    print("  'full' - Toggle between snippet and full content view")

    last_query = None
    last_bm25_results = None
    last_sbert_results = None
    show_full_content = False

    while True:
        # Get user input
        user_input = input("\nEnter query or command: ").strip()

        # Check for exit command
        if user_input.lower() == 'exit':
            print("Exiting the search system. Goodbye!")
            break

        # Check for view command
        if user_input.lower().startswith('view '):
            try:
                result_num = int(user_input.split()[1])
                if last_sbert_results and 1 <= result_num <= len(last_sbert_results):
                    doc = last_sbert_results[result_num - 1]['doc']
                    print(f"\nFull content of result {result_num}:")
                    print(f"Publication: {doc['publication']}")
                    print(f"Date: {doc['date']}")
                    print(f"Author: {doc['author']}")
                    print(f"Content: {doc['contents']}")
                else:
                    print("Invalid result number or no results available.")
            except (IndexError, ValueError):
                print("Invalid view command. Use 'view N' where N is the result number.")
            continue

        # Check for BM25 results command
        if user_input.lower() == 'bm25':
            if last_bm25_results:
                print(f"\nBM25 results for query: '{last_query}'")
                for i, doc in enumerate(last_bm25_results, 1):
                    snippet = doc["contents"][:200] + "..." if len(doc["contents"]) > 200 else doc["contents"]
                    print(f"\nResult {i}:")
                    print(f"Publication: {doc['publication']}")
                    print(f"Date: {doc['date']}")
                    print(f"Author: {doc['author']}")
                    print(f"Content snippet: {snippet}")
            else:
                print("No BM25 results available. Please run a search first.")
            continue

        # Check for full content toggle command
        if user_input.lower() == 'full':
            show_full_content = not show_full_content
            print(f"Full content view: {'ON' if show_full_content else 'OFF'}")
            if last_sbert_results:
                print_results(last_sbert_results, show_full_content)
            continue

        # Skip empty queries
        if not user_input:
            print("Query cannot be empty. Please try again.")
            continue

        try:
            # Apply simplified spell checking to handle typos
            corrected_query, was_corrected, corrections = spell_checker.correct_spelling(user_input)

            # Inform user if query was corrected
            if was_corrected:
                correction_details = ", ".join(
                    [f"'{word}' → '{correction}'" for word, correction in corrections.items()])
                print(f"\nAutomatically corrected query: '{corrected_query}' (Fixed: {correction_details})")
                query = corrected_query
            else:
                query = user_input

            # Save the query for later use
            last_query = query

            # Stage 1: BM25 retrieval
            print(f"\nSearching for: '{query}'")
            bm25_top_k = 10  # Number of documents to retrieve with BM25

            top_k_docs = bm25_algorithm(docs, query, top_k=bm25_top_k)
            last_bm25_results = top_k_docs

            if not top_k_docs:
                print("No results found. Please try a different query.")
                continue

            # Stage 2: SBERT re-ranking
            print("Re-ranking results with SBERT...")
            reranked_results = sbert_rerank(top_k_docs, query)
            last_sbert_results = reranked_results

            # Print results
            print_results(reranked_results, show_full_content)

        except Exception as e:
            print(f"An error occurred during search: {str(e)}")
            print("Please try a different query or check your input.")


if __name__ == "__main__":
    main()
