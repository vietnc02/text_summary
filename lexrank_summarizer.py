import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def lexrank_summarizer(text, summary_ratio=0.3):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # Tách câu
    sentences = [sent.text.strip() for sent in doc.sents]
    if len(sentences) < 2:
        return text, len(text.split()), len(text.split()), 0, text, {}

    # Vector hóa TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Tính ma trận tương đồng cosine
    cosine_matrix = cosine_similarity(tfidf_matrix)

    # Xây dựng đồ thị và áp dụng PageRank
    nx_graph = nx.from_numpy_array(cosine_matrix)
    scores = nx.pagerank(nx_graph)

    # Sắp xếp câu theo điểm PageRank
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    # Chọn các câu quan trọng nhất
    num_sentences = max(1, int(len(sentences) * summary_ratio))
    selected_sentences = [ranked_sentences[i][1] for i in range(num_sentences)]

    # Gộp các câu lại thành bản tóm tắt
    sorted_summary = sorted(selected_sentences, key=lambda s: sentences.index(s))
    summary_text = " ".join(sorted_summary)

    # Tô sáng các câu được chọn
    highlighted_text = text
    for sentence in selected_sentences:
        highlighted_text = highlighted_text.replace(sentence, f'<span class="highlight">{sentence}</span>')

    # Thông tin bổ sung
    summary_data = {
        'cosine_matrix': cosine_matrix,
        'pagerank_scores': scores,
        'ranked_sentences': ranked_sentences,
    }

    return summary_text, text, len(text.split()), len(summary_text.split()), 0, highlighted_text, summary_data

def main():
    while True:
        print("\nOptions:")
        print("1. Enter text for summarization")
        print("2. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            # Nhập văn bản từ người dùng
            text = input("\nEnter your text: ")
            
            # Nhập tỷ lệ tóm tắt từ người dùng
            while True:
                try:
                    summary_ratio = float(input("Enter summary ratio (e.g., 0.3 for 30%): "))
                    if 0 < summary_ratio <= 1:
                        break  # Đầu vào hợp lệ, thoát khỏi vòng lặp
                    else:
                        print("Please enter a number between 0 and 1.")
                except ValueError:
                    print("Invalid input. Please enter a valid number between 0 and 1.")
            
            summary_text, original_text, original_len, summary_len, _, highlighted_text, summary_data = lexrank_summarizer(text, summary_ratio)

            # Hiển thị kết quả xếp hạng các câu
            print("\n--- Sentence Rankings ---")
            for rank, (score, sentence) in enumerate(summary_data['ranked_sentences'], 1):
                print(f"Rank {rank}: (Score: {score:.4f}) {sentence}")

            # Hiển thị kết quả tóm tắt
            print("\n--- Summary ---")
            print(summary_text)
            print(f"\nOriginal Length: {original_len} words")
            print(f"Summary Length: {summary_len} words")
        elif choice == "2":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
