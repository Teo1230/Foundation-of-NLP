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
from collections import Counter
import matplotlib.pyplot as plt
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Apply a consistent style
plt.style.use('seaborn-muted')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

# Function to load text

def load_text():
    path = os.getcwd()
    path_centuries = os.path.join(path, 'century')
    txt = {}

    for century in os.listdir(path_centuries):
        path_century = os.path.join(path_centuries, century)
        txt[century] = []

        for file in os.listdir(path_century):
            file_path = os.path.join(path_century, file)
            if file.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if text:
                    txt[century].append(text)

    return txt

# Function to generate plots
def generate_plots(txt):
    # Helper function for annotations
    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.0f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 5),  # Offset text by 5 pixels
                         textcoords="offset points",
                         ha='center', va='bottom')

    # Plot 1: Distribution of Texts Across Centuries
    counts = {century: len(texts) for century, texts in txt.items()}
    rects = plt.bar(counts.keys(), counts.values(), color='skyblue')
    plt.xlabel('Century')
    plt.ylabel('Number of Texts')
    plt.title('Distribution of Texts Across Centuries')
    annotate_bars(rects)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('text_distribution.png')
    plt.close()

    # Plot 2: Average Word Count Per Century
    avg_words = {century: sum(len(text.split()) for text in texts) / len(texts) for century, texts in txt.items()}
    rects = plt.bar(avg_words.keys(), avg_words.values(), color='coral')
    plt.xlabel('Century')
    plt.ylabel('Average Word Count')
    plt.title('Average Word Count Per Century')
    annotate_bars(rects)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('avg_word_count.png')
    plt.close()

    # Plot 3: Word Cloud for Each Century
    for century, texts in txt.items():
        combined_text = ' '.join(texts)
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(combined_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Century {century}')
        plt.savefig(f'word_cloud_{century}.png')
        plt.close()



    # Combine all texts and count word frequencies, filtering for words with length > 2
    all_texts = ' '.join([' '.join(texts) for texts in txt.values()])
    word_counts = Counter(word for word in all_texts.split() if len(word) > 3)

    # Get the 15 most common words
    common_words = word_counts.most_common(15)
    words, counts = zip(*common_words)

    # Create the bar plot
    rects = plt.bar(words, counts, color='mediumpurple')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Common Words Across Centuries')

    # Annotate bars with their heights
    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # Offset text slightly above the bar
                         textcoords="offset points",
                         ha='center', va='bottom')

    annotate_bars(rects)

    # Add grid lines and save the plot
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('common_words.png')
    plt.close()

    # Plot 5: Text Length Distribution
    lengths = [len(text.split()) for texts in txt.values() for text in texts]
    plt.hist(lengths, bins=30, color='teal', edgecolor='black')
    plt.xlabel('Text Length (Words)')
    plt.ylabel('Frequency')
    plt.title('Text Length Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('text_length_distribution.png')
    plt.close()

    # Plot 6: Vocabulary Size Per Century
    vocab_sizes = {century: len(set(' '.join(texts).split())) for century, texts in txt.items()}
    rects = plt.bar(vocab_sizes.keys(), vocab_sizes.values(), color='gold')
    plt.xlabel('Century')
    plt.ylabel('Vocabulary Size')
    plt.title('Vocabulary Size Per Century')
    annotate_bars(rects)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('vocab_size.png')
    plt.close()

    # Plot 7: POS Tag Distribution Per Century
    pos_counts = {}
    for century, texts in txt.items():
        all_tags = [tag for text in texts for _, tag in pos_tag(word_tokenize(text))]
        pos_counts[century] = Counter(all_tags)
    for century, counts in pos_counts.items():
        tags, frequencies = zip(*counts.items())
        rects = plt.bar(tags, frequencies, color='orchid')
        plt.xlabel('POS Tags')
        plt.ylabel('Frequency')
        plt.title(f'POS Distribution for Century {century}')
        plt.xticks(rotation=45, ha='right')
        annotate_bars(rects)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f'pos_distribution_{century}.png')
        plt.close()

    # Plot 8: Sentiment Distribution Per Century
    sentiments = {century: [TextBlob(text).sentiment.polarity for text in texts] for century, texts in txt.items()}
    for century, scores in sentiments.items():
        plt.hist(scores, bins=30, alpha=0.7, label=f'{century}', edgecolor='black')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.title('Sentiment Distribution Per Century')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('sentiment_distribution.png')
    plt.close()

    # New Plot 9: Temporal Word Trends
    word = 'example'  # Replace with the desired word
    word_freq = {century: sum(text.lower().count(word) for text in texts) for century, texts in txt.items()}
    plt.plot(word_freq.keys(), word_freq.values(), marker='o', color='darkorange')
    plt.xlabel('Century')
    plt.ylabel('Frequency')
    plt.title(f'Temporal Trend of Word: "{word}"')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'temporal_trend_{word}.png')
    plt.close()

    # New Plot 10: Unique Word Growth Over Centuries
    centuries = sorted(txt.keys())
    unique_words = set()
    growth = []
    for century in centuries:
        century_text = ' '.join(txt[century])
        unique_words.update(century_text.split())
        growth.append(len(unique_words))
    plt.plot(centuries, growth, marker='o', color='seagreen')
    plt.xlabel('Century')
    plt.ylabel('Cumulative Unique Words')
    plt.title('Unique Word Growth Over Centuries')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('unique_word_growth.png')
    plt.close()

    # New Plot 11: Bigram Frequency Heatmap
    bigram_dist = FreqDist(bigrams(all_texts.split()))
    common_bigrams = bigram_dist.most_common(15)
    bigrams_list, counts = zip(*common_bigrams)
    bigrams_str = [' '.join(b) for b in bigrams_list]
    plt.bar(bigrams_str, counts, color='dodgerblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Bigrams')
    plt.ylabel('Frequency')
    plt.title('Most Common Bigrams')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('bigram_frequency.png')
    plt.close()

    # New Plot 12: Readability Scores
    readability = {century: sum(flesch_reading_ease(text) for text in texts) / len(texts) for century, texts in txt.items()}
    rects = plt.bar(readability.keys(), readability.values(), color='slateblue')
    plt.xlabel('Century')
    plt.ylabel('Readability Score')
    plt.title('Readability Scores Per Century')
    annotate_bars(rects)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('readability_scores.png')
    plt.close()

# Example usage
txt = load_text()
generate_plots(txt)

