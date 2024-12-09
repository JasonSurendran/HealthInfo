import os
import math
from collections import Counter
import shutil


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
    :return: List of document contents as strings and their filenames.
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


def ndcg_at_k(relevance_scores, k=10):
    """
    Calculate nDCG@k.
    :param relevance_scores: List of relevance scores.
    :param k: Rank cutoff.
    :return: nDCG@k score.
    """
    relevance_scores = relevance_scores[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance[:k]))
    return dcg / idcg if idcg > 0 else 0


def save_ndcg_to_file(output_file, ndcg_score):
    """
    Save the nDCG@10 score to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"nDCG@10: {ndcg_score:.4f}\n")
    print(f"nDCG@10 score saved to {output_file}.")


def run(src_directory, dest_directory):
    # Load documents
    documents, filenames = load_documents_from_directory(src_directory)

    # Query
    query = ["diet", "exercise", "health"]

    # User-provided relevance scores (example: ground truth relevance for each document)
    relevance_scores = [10, 8, 6, 5, 3, 10, 8, 6, 5, 3, 10, 8, 6, 5, 3]  # Adjust this to reflect actual relevance of documents

    # Initialize BM25
    bm25 = BM25(documents)

    # Compute scores
    scores = bm25.compute_bm25(query)

    # Rank documents
    ranked_docs = sorted(zip(scores, filenames, documents, relevance_scores), reverse=True, key=lambda x: x[0])

    print("Ranked Documents:")
    for score, filename, doc, rel in ranked_docs:
        print(f"Score: {score:.4f}, Filename: {filename}, Relevance: {rel}")

    # Extract ranked relevance scores for nDCG calculation
    ranked_relevance = [rel for _, _, _, rel in ranked_docs]

    # Calculate nDCG@10
    ndcg_score = ndcg_at_k(ranked_relevance, k=10)
    print(f"nDCG@10: {ndcg_score:.4f}")

    # Save nDCG score to file
    ndcg_output_file = "output_files/ndcg_score.txt"
    save_ndcg_to_file(ndcg_output_file, ndcg_score)

    # Threshold for filtering
    threshold = 1.5

    # Copy relevant documents
    copy_relevant_documents(filenames, scores, threshold, src_directory, dest_directory)

    print(f"Filtering complete. Documents with scores >= {threshold} moved to {dest_directory}.")

