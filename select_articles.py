import os
import math
from collections import Counter
import shutil


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.avg_doc_len = sum(len(doc.split()) for doc in documents) / len(documents)
        self.doc_lens = [len(doc.split()) for doc in documents]
        self.doc_freqs = self._compute_document_frequencies()

    def _compute_document_frequencies(self):
        df = Counter()
        for doc in self.documents:
            unique_terms = set(doc.split())
            for term in unique_terms:
                df[term] += 1
        return df

    def _compute_idf(self, term):
        n = len(self.documents)
        doc_freq = self.doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0
        return math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def compute_bm25(self, query):
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
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename, score in zip(filenames, scores):
        if score >= threshold:
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dest_path)


def ndcg_at_k(relevance_scores, k=10):
    relevance_scores = relevance_scores[:k]
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance[:k]))
    return dcg / idcg if idcg > 0 else 0


def save_ndcg_to_file(output_file, ndcg_score):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"nDCG@10: {ndcg_score:.4f}\n")


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

    # Extract ranked relevance scores for nDCG calculation
    ranked_relevance = [rel for _, _, _, rel in ranked_docs]

    # Calculate nDCG@10
    ndcg_score = ndcg_at_k(ranked_relevance, k=10)

    # Save nDCG score to file
    ndcg_output_file = "output_files/ndcg_score.txt"
    save_ndcg_to_file(ndcg_output_file, ndcg_score)

    # Threshold for filtering
    threshold = 1.5

    # Copy relevant documents
    copy_relevant_documents(filenames, scores, threshold, src_directory, dest_directory)

