import os
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Top 100 most common English words
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on",
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we",
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any",
    "these", "give", "day", "most", "us", "is", "are", "more", "include", "such", "may", 
    "less", "youre", "keep", "should", "need", "best", "down", "type", "find", "too", "feel", 
    "sure", "increase"
}


def load_documents(directory):
    """
    Load all .txt documents from the specified directory.
    """
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read().lower()
    return documents


def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase and removing special characters.
    """
    return re.sub(r'[^a-z0-9\s]', '', text)


def extract_common_words(doc1, doc2):
    """
    Find common individual words between two documents.
    """
    # Tokenize and count word frequencies in both documents
    words1 = Counter(doc1.split())
    words2 = Counter(doc2.split())

    # Find common words and their combined frequencies
    common_words = words1 & words2  # Intersection of both Counters
    return common_words


def filter_common_words(word_counts):
    """
    Filter out the top 100 most common English words.
    """
    return {word: count for word, count in word_counts.items() if word not in COMMON_WORDS}


def calculate_cosine_similarity(doc1, doc2):
    """
    Calculate cosine similarity between two documents.
    """
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(vectors)[0][1]
    return similarity


def find_common_words_and_similarity(documents):
    """
    Find common individual words and calculate cosine similarity across all document pairs.
    """
    filenames = list(documents.keys())
    results = []
    all_words = Counter()

    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            doc1 = preprocess_text(documents[filenames[i]])
            doc2 = preprocess_text(documents[filenames[j]])

            # Calculate cosine similarity
            similarity = calculate_cosine_similarity(doc1, doc2)

            # Extract and filter common words
            common_words = extract_common_words(doc1, doc2)
            filtered_words = filter_common_words(common_words)
            all_words.update(filtered_words)

            # Add pair similarity to results
            results.append({
                "Document 1": filenames[i],
                "Document 2": filenames[j],
                "Cosine Similarity": similarity,
                "Common Words": filtered_words
            })

    # Sort words by frequency in descending order
    ranked_words = all_words.most_common()
    return results, ranked_words


def save_common_words_to_file(common_words, output_file):
    """
    Save the common words to a text file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for word, freq in common_words:
            file.write(f"{word}\t{freq}\n")


def save_similarity_results(results, output_file):
    """
    Save document similarity results to a file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(f"{result['Document 1']} - {result['Document 2']}: {result['Cosine Similarity']:.4f}\n")


def run(directory, words_output_file, similarity_output_file):
    # Load documents and find common words and similarity
    documents = load_documents(directory)
    similarity_results, common_words_ranked = find_common_words_and_similarity(documents)

    # Save the filtered common words and similarity results
    save_common_words_to_file(common_words_ranked, words_output_file)
    save_similarity_results(similarity_results, similarity_output_file)

    print(f"Filtered common words saved to {words_output_file}.")
    print(f"Document similarity results saved to {similarity_output_file}.")