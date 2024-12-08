import os
import math
from collections import Counter

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        """
        Initialize BM25 with a list of documents.
        :param documents: List of strings (each string is a document).
        :param k1: BM25 k1 parameter.
        :param b: BM25 b parameter.
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_len = sum(len(doc.split()) for doc in documents) / len(documents)
        self.doc_lens = [len(doc.split()) for doc in documents]
        self.doc_freqs = self._compute_document_frequencies()

    def _compute_document_frequencies(self):
        """
        Calculate document frequency for each term across all documents.
        """
        df = Counter()
        for doc in self.documents:
            unique_terms = set(doc.split())
            for term in unique_terms:
                df[term] += 1
        return df

    def _compute_idf(self, term):
        """
        Calculate Inverse Document Frequency (IDF) for a term.
        """
        n = len(self.documents)
        doc_freq = self.doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0
        return math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def compute_bm25(self, query):
        """
        Calculate BM25 scores for the query.
        :param query: List of query terms.
        :return: List of BM25 scores for each document.
        """
        scores = []
        for idx, doc in enumerate(self.documents):
            score = 0
            doc_len = self.doc_lens[idx]
            term_freqs = Counter(doc.split())
            for term in query:
                idf = self._compute_idf(term)
                tf = term_freqs[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores

def load_documents_from_directory(directory):
    """
    Load all text files from a given directory.
    :param directory: Path to the directory containing text files.
    :return: List of document contents as strings.
    """
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                filenames.append(filename)
    return documents, filenames

def copy_relevant_documents(filenames, scores, threshold, src_dir, dest_dir):
    """
    Copy documents with scores above the threshold to the destination directory.
    :param filenames: List of document filenames.
    :param scores: List of BM25 scores for each document.
    :param threshold: Threshold for BM25 scores.
    :param src_dir: Source directory containing documents.
    :param dest_dir: Destination directory for filtered documents.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename, score in zip(filenames, scores):
        if score >= threshold:
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dest_path)
            print(f"Copied: {filename} with score {score:.4f}")
            
# Example Usage
if __name__ == "__main__":
    # Directories
    src_directory = "./documents"
    dest_directory = "./documents_filtered"

    # Load documents
    documents, filenames = load_documents_from_directory(src_directory)
    
    # Query
    query = ["diet", "exercise", "health"]

    # Initialize BM25
    bm25 = BM25(documents)

    # Compute scores
    scores = bm25.compute_bm25(query)

    # Threshold for filtering
    threshold = 2.0

    # Rank documents
    ranked_docs = sorted(zip(scores, filenames, documents), reverse=True, key=lambda x: x[0])

    print("Ranked Documents:")
    for score, filename, doc in ranked_docs:
        print(f"Score: {score:.4f}, Filename: {filename}")

    # Copy relevant documents
    copy_relevant_documents(filenames, scores, threshold, src_directory, dest_directory)

    print(f"Filtering complete. Documents with scores >= {threshold} moved to {dest_directory}.")