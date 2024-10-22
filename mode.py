import os


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


import re
import string
import spacy

# Load the Romanian spaCy model
nlp_ro = spacy.load("ro_core_news_sm")
nlp_ro.max_length = 2000000  # Or any value greater than your maximum text length


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


import copy

txt = load_text()
st = sorted(txt)
t = copy.deepcopy(txt)

texts, labels = [], []

for century in st:
    print(f"Century: {century} -> {len(txt[century])}")
    for data in txt[century]:
        # print(data[:10000])
        # text = prep(data[:10000]) -> 0.5 random , 0.9473684210526315 full data
        text = prep(data)
        texts.append(text)
        labels.append(century)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 3: Train-test split with stratification
num_classes = len(label_encoder.classes_)
test_size = max(0.2, (num_classes / len(texts)))  # Ensure test size accommodates all classes
print(test_size, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    encoded_labels,
    test_size=test_size,
    random_state=42,
    stratify=encoded_labels
)

# Step 4: Vectorization
vectorizer = TfidfVectorizer()

# Step 5: Initialize models and their parameter grids for tuning
logistic_params = {
    'logisticregression__C': [0.1, 1, 10],
    'logisticregression__solver': ['liblinear'],
    'logisticregression__penalty': ['l1'],
    'logisticregression__dual': [False]
}

logistic_params_lbfgs = {
    'logisticregression__C': [0.1, 1, 10],
    'logisticregression__solver': ['lbfgs'],
    'logisticregression__penalty': ['l2'],
}

models_and_params = {
    "Random Forest": (RandomForestClassifier(), {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20, 30],
        'randomforestclassifier__min_samples_split': [2, 5, 10]
    }),
    "SVM": (SVC(), {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }),
    "Logistic Regression (Liblinear)": (LogisticRegression(max_iter=1000), logistic_params),
    "Logistic Regression (LBFGS)": (LogisticRegression(max_iter=1000), logistic_params_lbfgs),
    "Naive Bayes": (MultinomialNB(), {
        'multinomialnb__alpha': [0.1, 1.0, 2.0]
    })
}

# Step 6: Train and evaluate models with hyperparameter tuning
for model_name, (model, params) in models_and_params.items():
    print(f"Training {model_name} with GridSearchCV...")
    pipeline = make_pipeline(vectorizer, model)
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} Accuracy: {accuracy:.4f}\n")
