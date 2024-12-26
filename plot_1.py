import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize, FreqDist
from nltk.util import bigrams
from textblob import TextBlob
from textstat import flesch_reading_ease
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
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

def generate_plots(txt):
    # Plot 1: Distribution of Texts Across Centuries
    counts = {century: len(texts) for century, texts in txt.items()}
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('Century')
    plt.ylabel('Number of Texts')
    plt.title('Distribution of Texts Across Centuries')
    plt.savefig('text_distribution.png')
    plt.close()

    # Plot 2: Average Word Count Per Century
    avg_words = {century: sum(len(text.split()) for text in texts) / len(texts) for century, texts in txt.items()}
    plt.bar(avg_words.keys(), avg_words.values())
    plt.xlabel('Century')
    plt.ylabel('Average Word Count')
    plt.title('Average Word Count Per Century')
    plt.savefig('avg_word_count.png')
    plt.close()

    # Plot 3: Word Cloud for Each Century
    for century, texts in txt.items():
        combined_text = ' '.join(texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Century {century}')
        plt.savefig(f'word_cloud_{century}.png')
        plt.close()

    # Plot 4: Most Common Words Across Centuries
    all_texts = ' '.join([' '.join(texts) for texts in txt.values()])
    word_counts = Counter(all_texts.split())
    common_words = word_counts.most_common(15)
    words, counts = zip(*common_words)
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words Across Centuries')
    plt.savefig('common_words.png')
    plt.close()

    # Plot 5: Text Length Distribution
    lengths = [len(text.split()) for texts in txt.values() for text in texts]
    plt.hist(lengths, bins=30)
    plt.xlabel('Text Length (Words)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.savefig('text_length_distribution.png')
    plt.close()

    # Plot 6: Vocabulary Size Per Century
    vocab_sizes = {century: len(set(' '.join(texts).split())) for century, texts in txt.items()}
    plt.bar(vocab_sizes.keys(), vocab_sizes.values())
    plt.xlabel('Century')
    plt.ylabel('Vocabulary Size')
    plt.title('Vocabulary Size Per Century')
    plt.savefig('vocab_size.png')
    plt.close()

    # Plot 7: Average Sentence Length Per Century
    avg_sentences = {}
    for century, texts in txt.items():
        sentence_lengths = [len(re.split(r'[.!?]', text)) for text in texts]
        avg_sentences[century] = sum(sentence_lengths) / len(sentence_lengths)
    plt.bar(avg_sentences.keys(), avg_sentences.values())
    plt.xlabel('Century')
    plt.ylabel('Average Sentence Length')
    plt.title('Average Sentence Length Per Century')
    plt.savefig('avg_sentence_length.png')
    plt.close()

    # Plot 8: Stopword Frequency Analysis
    stop_words = set(stopwords.words('english'))
    stop_counts = {century: sum(word in stop_words for text in texts for word in text.split()) for century, texts in txt.items()}
    plt.bar(stop_counts.keys(), stop_counts.values())
    plt.xlabel('Century')
    plt.ylabel('Stopword Frequency')
    plt.title('Stopword Frequency Analysis')
    plt.savefig('stopword_frequency.png')
    plt.close()

    # Plot 9: Temporal Word Trends
    word = 'example'  # Replace 'example' with the word you want to track
    word_freq = {century: sum(text.count(word) for text in texts) for century, texts in txt.items()}
    plt.plot(word_freq.keys(), word_freq.values(), marker='o')
    plt.xlabel('Century')
    plt.ylabel('Frequency')
    plt.title(f'Temporal Trend of Word: "{word}"')
    plt.savefig(f'temporal_trend_{word}.png')
    plt.close()

    # Plot 10: POS Tag Distribution Per Century
    pos_counts = {}
    for century, texts in txt.items():
        all_tags = [tag for text in texts for _, tag in pos_tag(word_tokenize(text))]
        pos_counts[century] = Counter(all_tags)
    for century, counts in pos_counts.items():
        plt.bar(counts.keys(), counts.values())
        plt.xlabel('POS Tags')
        plt.ylabel('Frequency')
        plt.title(f'POS Distribution for Century {century}')
        plt.savefig(f'pos_distribution_{century}.png')
        plt.close()

    # Plot 11: Bigram Frequency
    bigram_dist = FreqDist(bigrams(all_texts.split()))
    common_bigrams = bigram_dist.most_common(15)
    bigrams_list, counts = zip(*common_bigrams)
    plt.bar([' '.join(b) for b in bigrams_list], counts)
    plt.xticks(rotation=45)
    plt.title('Most Common Bigrams')
    plt.savefig('bigram_frequency.png')
    plt.close()

    # Plot 12: Sentiment Distribution Per Century
    sentiments = {century: [TextBlob(text).sentiment.polarity for text in texts] for century, texts in txt.items()}
    for century, scores in sentiments.items():
        plt.hist(scores, bins=30, alpha=0.7, label=f'{century}')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution Per Century')
    plt.legend()
    plt.savefig('sentiment_distribution.png')
    plt.close()

    # Plot 13: Unique Word Growth Over Centuries
    centuries = sorted(txt.keys())
    unique_words = set()
    growth = []
    for century in centuries:
        century_text = ' '.join(txt[century])
        unique_words.update(century_text.split())
        growth.append(len(unique_words))
    plt.plot(centuries, growth, marker='o')
    plt.xlabel('Century')
    plt.ylabel('Cumulative Unique Words')
    plt.title('Unique Word Growth Over Centuries')
    plt.savefig('unique_word_growth.png')
    plt.close()

    # Plot 14: Readability Scores
    readability = {century: sum(flesch_reading_ease(text) for text in texts) / len(texts) for century, texts in txt.items()}
    plt.bar(readability.keys(), readability.values())
    plt.xlabel('Century')
    plt.ylabel('Readability Score')
    plt.title('Readability Scores Per Century')
    plt.savefig('readability_scores.png')
    plt.close()

    # Plot 15: N-Gram Diversity
    n = 3
    diversity = {century: len(set(nltk.ngrams(' '.join(texts).split(), n))) for century, texts in txt.items()}
    plt.bar(diversity.keys(), diversity.values())
    plt.xlabel('Century')
    plt.ylabel(f'Unique {n}-grams')
    plt.title(f'{n}-Gram Diversity Per Century')
    plt.savefig(f'ngram_diversity_{n}.png')
    plt.close()

# Example usage
txt = load_text()  # Replace with your load_text function
generate_plots(txt)

