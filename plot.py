import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from collections import Counter
from textblob import TextBlob
import numpy as np

# Set Seaborn theme for better visuals
sns.set_theme(style="whitegrid")

def load_text():
    path = os.getcwd()  # Current working directory
    path_centuries = os.path.join(path, 'century')  # Path to 'century' folder
    txt = {}  # Dictionary to store raw text by century

    # Loop through each century folder
    for century in os.listdir(path_centuries):
        path_century = os.path.join(path_centuries, century)
        txt[century] = []

        # Loop through PDF files in each century folder
        for pdf_file in os.listdir(path_century):
            file_path = os.path.join(path_century, pdf_file)

            # Check if the file is a PDF
            if pdf_file.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:

                    text = f.read()

                # Append the raw text to the century's list
                if text:  # Ensure text is not empty
                    txt[century].append(text)

    return txt

def create_useful_plots(txt):
    # Create the images folder if it doesn't exist
    images_folder = "images"
    os.makedirs(images_folder, exist_ok=True)

    # Analyze data
    centuries = list(txt.keys())
    document_counts = [len(txt[century]) for century in centuries]
    word_counts = [
        sum(len(document.split()) for document in txt[century]) for century in centuries
    ]

    # Plot 1: Number of documents per century
    plt.figure(figsize=(10, 6))
    sns.barplot(x=centuries, y=document_counts, palette="Blues_d")
    plt.title("Number of Documents per Century")
    plt.xlabel("Century")
    plt.ylabel("Document Count")
    plt.savefig(os.path.join(images_folder, "documents_per_century.png"))
    plt.close()

    # Plot 2: Average word count per document by century
    avg_word_counts = [
        word_counts[i] / document_counts[i] if document_counts[i] > 0 else 0
        for i in range(len(centuries))
    ]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=centuries, y=avg_word_counts, palette="Greens_d")
    plt.title("Average Word Count per Document by Century")
    plt.xlabel("Century")
    plt.ylabel("Average Word Count")
    plt.savefig(os.path.join(images_folder, "avg_word_count_per_document.png"))
    plt.close()

    # Plot 3: Distribution of word counts (violin plot)
    word_count_data = [
        [len(document.split()) for document in txt[century]] for century in centuries
    ]
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=word_count_data, scale="width", inner="quartile", palette="pastel")
    plt.title("Distribution of Word Counts by Century")
    plt.xlabel("Century")
    plt.ylabel("Word Count")
    plt.xticks(range(len(centuries)), centuries)
    plt.savefig(os.path.join(images_folder, "word_count_violin_plot.png"))
    plt.close()

    # Plot 4: Total word count by century
    plt.figure(figsize=(10, 6))
    sns.barplot(x=centuries, y=word_counts, palette="coolwarm")
    plt.title("Total Word Count by Century")
    plt.xlabel("Century")
    plt.ylabel("Total Word Count")
    plt.savefig(os.path.join(images_folder, "total_word_count_by_century.png"))
    plt.close()

    # Plot 5: Lexical Diversity by Century
    lexical_diversity = [
        len(set(" ".join(txt[century]).split())) / max(1, word_counts[i]) for i, century in enumerate(centuries)
    ]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=centuries, y=lexical_diversity, palette="Purples_d")
    plt.title("Lexical Diversity by Century")
    plt.xlabel("Century")
    plt.ylabel("Lexical Diversity (Unique Tokens/Total Tokens)")
    plt.savefig(os.path.join(images_folder, "lexical_diversity_by_century.png"))
    plt.close()

    # Plot 6: Sentiment Polarity by Century
    sentiment_polarity = [
        np.mean([TextBlob(doc).sentiment.polarity for doc in txt[century]]) for century in centuries
    ]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=centuries, y=sentiment_polarity, palette="RdYlGn")
    plt.title("Average Sentiment Polarity by Century")
    plt.xlabel("Century")
    plt.ylabel("Sentiment Polarity (-1 to 1)")
    plt.savefig(os.path.join(images_folder, "sentiment_polarity_by_century.png"))
    plt.close()

    # Plot 7: Word Cloud for Each Century
    for century in centuries:
        all_text = " ".join(txt[century])
        wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(all_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {century} Century")
        plt.savefig(os.path.join(images_folder, f"word_cloud_{century}.png"))
        plt.close()

    # # Plot 8: Embedding Visualization (t-SNE)
    # all_texts = [" ".join(txt[century]) for century in centuries]
    # word_counts_vectorized = [len(text.split()) for text in all_texts]  # Simplified example
    # if len(all_texts) > 1 and len(all_texts) > 2:  # Ensure enough samples for t-SNE
    #     perplexity = min(30, len(all_texts) - 1)  # Adjust perplexity to fit data
    #     embeddings = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(
    #         np.array(word_counts_vectorized).reshape(-1, 1)
    #     )
    #     plt.figure(figsize=(10, 6))
    #     sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=centuries, palette="tab10", legend="full")
    #     plt.title("t-SNE Embedding Visualization")
    #     plt.xlabel("Component 1")
    #     plt.ylabel("Component 2")
    #     plt.savefig(os.path.join(images_folder, "embedding_visualization_tsne.png"))
    #     plt.close()
    # else:
    #     print("Not enough data points for t-SNE visualization.")

# Example usage
txt = load_text()  # Uncomment this line when you have the 'century' folder structure
create_useful_plots(txt)

