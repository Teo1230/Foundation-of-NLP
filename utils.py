import os
import re
import spacy
import string

def load_text(dataset='train'):
    path = os.getcwd()  # Current working directory
    path_data = os.path.join(path, dataset)
    path_centuries = os.path.join(path_data, 'century')  # Path to 'century' folder
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

# Load the Romanian spaCy model
nlp_ro = spacy.load("ro_core_news_sm")
nlp_ro.max_length = 4000000  # Or any value greater than your maximum text length

def has_vowel(word):
    """Check if the word contains at least one vowel."""
    vowels = "aeiouăâî"
    return any(char in vowels for char in word)

def prep(text):
    # Lowercase the text
    text = text.lower()

    # Replace specified characters with Romanian diacritics
    text = text.replace("[", "ă")
    text = text.replace("{", "ă")
    text = text.replace("`", "â")
    text = text.replace("=", "ș")
    text = text.replace("\\", "ț")  # Ensure the backslash is handled correctly
    text = text.replace("]", "î")
    text = text.replace("}", "Î")
    text = text.replace("|", "Ț")
    text = text.replace("º", "ș")
    text = text.replace("§", "")
    text = text.replace("˘", "")

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Roman numerals (I to XII and beyond)
    roman_numeral_pattern = r'\b(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))\b'
    text = re.sub(roman_numeral_pattern, '', text)

    # Process the text with spaCy
    doc = nlp_ro(text)

    # Filter out unwanted tokens and lemmatize
    sanitized_tokens = []
    for token in doc:
        # Conditions to filter tokens
        if (token.is_alpha and not token.is_stop and
                len(token) > 1 and has_vowel(token.lemma_)):  # Check for vowel presence
            sanitized_tokens.append(token.lemma_)  # Use the lemma of the token

    # Join the sanitized tokens back into a single string
    sanitized_text = ' '.join(sanitized_tokens)

    return sanitized_text

