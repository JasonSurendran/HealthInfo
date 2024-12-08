import select_articles
import find_common_phrases
import display_dashboard

if __name__ == "__main__":
    select_articles.run("./documents", "./documents_filtered")
    find_common_phrases.run("./documents_filtered", "output_files/common_words_filtered.txt", "output_files/document_similarity.txt")
    display_dashboard.run("output_files/common_words_filtered.txt", "output_files/document_similarity.txt")