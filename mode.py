import os
import fitz  # PyMuPDF
import os
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import roner
import re
import joblib
from itertools import product
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier


# Extract text from a PDF file
def extract_text_from_pdf(pdf_file_path):
    text = ""
    try:
        with fitz.open(pdf_file_path) as pdf:
            # Start reading from page 10 (index 9)
            for page_num in range(0, pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text()  # Extract text from each page
    except Exception as e:
        print(f"Error reading {pdf_file_path}: {e}")
    return text.strip()


# Function to load text data from a folder structure
def load_text_data():
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
            if pdf_file.endswith('.pdf'):
                # Extract raw text from the PDF
                text = extract_text_from_pdf(file_path)

                # Append the raw text to the century's list
                if text:  # Ensure text is not empty
                    txt[century].append(text)

    return txt


# Ensure to install nltk and download the stopwords: nltk.download('stopwords')
stop_words = set(stopwords.words('romanian'))  # Use Romanian stopwords


def prep(text):
    # Lowercase the text
    text = text.lower()
    # Replace specified characters
    text = text.replace("[", "ă")
    text = text.replace("`", "â")
    text = text.replace("=", "ș")
    text = text.replace("\\", "ț")  # Ensure the backslash is handled correctly
    text = text.replace("]", "î")
    text = text.replace("}", "Î")
    text = text.replace("|", "Ț")
    text = text.replace("º", "ș")
    text = text.replace("§", "")

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove Roman numerals (I, II, III, IV, V, VI, VII, VIII, IX, X, and beyond)
    text = re.sub(r'\b(M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|[IVXLCDM]+)\b', '', text,
                  flags=re.IGNORECASE)

    # Remove Latin numbers (0-9)
    text = re.sub(r'\b[0-9]+\b', '', text)

    # Remove stopwords
    filtered_words = [word for word in text.split() if word not in stop_words]

    # Join words after the 500th word
    result_text = ' '.join(filtered_words[500:])

    # Optionally, strip leading/trailing spaces
    return result_text.strip()


nlp_ro = spacy.load("ro_core_news_sm")
nlp_ro.max_length = 2000000  # Or any value greater than your maximum text length

ner = roner.NER()


def vectorize_text_data(txt, max_features=1000):
    corpus = []
    labels = []
    named_entities = []
    for century, texts in txt.items():
        for text in texts:
            named_entities_text = ner(text)
            named_entities.append(' '.join([entity['text'] for entity in named_entities_text]))
            corpus.append(text)
            labels.append(century)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    X_named_entities = vectorizer.transform(named_entities)
    X_combined = np.hstack((X.toarray(), X_named_entities.toarray()))
    y = np.array(labels)
    return X_combined, y, vectorizer


def train_classifier(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    if model == 'LinearSVC':
        clf = LinearSVC(C=0.5, loss='squared_hinge', penalty='l2', multi_class='ovr', random_state=15)
    elif model == 'KNeighbors':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif model == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(max_depth=3)
    elif model == 'MultinomialNB':
        clf = MultinomialNB()
    elif model == 'GaussianNB':
        clf = GaussianNB()
    else:
        raise ValueError("Invalid model name!")

    clf.fit(X_train, y_train)

    return clf, X_test, y_test


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Weighted accuracy:", weighted_accuracy(y_test, y_pred))


def weighted_accuracy(y_true, y_pred):
    distances = np.abs(y_true.astype(int) - y_pred.astype(int))
    partial_quotients = 1 / 2 ** distances
    return np.sum(partial_quotients) / len(y_true)


def predict_century(new_text, vectorizer, clf):
    new_text = preprocess_text_ro(new_text)
    named_entities_new = ner(new_text)
    named_entities_text = ' '.join([entity['text'] for entity in named_entities_new])
    new_text_vectorized = vectorizer.transform([new_text])
    named_entities_vectorized = vectorizer.transform([named_entities_text])
    combined_features = np.hstack((new_text_vectorized.toarray(), named_entities_vectorized.toarray()))
    predicted_century = clf.predict(combined_features)
    return predicted_century[0]


if __name__ == "__main__":
    # Load and extract raw text data from PDF files
    text_data_by_century = load_text_data()
    st = sorted(text_data_by_century)

    for century in st:
        print(f"Century: {century}")
        texts = text_data_by_century[century]
        print(len(texts))
        for data in texts:
            # print(data['author'])
            text = prep(data)
            data = text

        X, y, vectorizer = vectorize_text_data(text_data_by_century)

        clf, X_test, y_test = train_classifier(X, y, 'LinearSVC')

        model_filename = 'text_classification_model_knn_7_sample.pkl'
        vectorizer_filename = 'tfidf_vectorizer_model_knn_7_sample.pkl'

        joblib.dump(clf, model_filename)

        joblib.dump(vectorizer, vectorizer_filename)

        print("Model and vectorizer saved successfully.")

        evaluate_model(clf, X_test, y_test)

        example = """ ce se petrece... Şi ne pomenim într-una din zile că părintele vine la şcoală şi ne aduce un scaun nou şi
    lung, şi
    după ce-a întrebat de dascăl, care cum ne purtăm, a stat puţin pe gânduri, apoi a pus nume scaunului -
    Calul Balan
    şi l-a lăsat în şcoală.
    În altă zi ne trezim că iar vine părintele la şcoală, cu moş Fotea, cojocarul satului, care ne aduce, dar
    de şcoală
    nouă, un drăguţ de biciuşor de curele, împletit frumos, şi părintele îi pune nume - Sfântul Nicolai,
    după cum este
    şi hramul bisericii din Humuleşti... Apoi pofteşte pe moş Fotea că, dacă i-or mai pica ceva curele
    bune, să mai facă
    aşa, din când în când, câte unul, şi ceva mai grosuţ, dacă se poate... Bădiţa Vasile a zâmbit atunci,
    iară noi, şcolarii,
    am rămas cu ochii holbaţi unii la alţii. Şi a pus părintele pravilă şi a zis că în toată sâmbăta să se
    procitească băieţii
    şi fetele, adică să asculte dascălul pe fiecare de tot ce-a învăţat peste săptămână; şi câte greşeli va
    face să i le
    însemne cu cărbune pe ceva, iar la urma urmelor, de fiecare greşeală să-i ardă şcolarului câte un
    sfânt-Nicolai.
    Atunci copila părintelui, cum era sprinţară şi plină de incuri, a bufnit în râs. Păcatul ei, sărmana! - Ia,
    """
        loaded_model = joblib.load(model_filename)

        loaded_vectorizer = joblib.load(vectorizer_filename)

        predicted_century = predict_century(example, loaded_vectorizer, loaded_model)
        print("Predicted century:", predicted_century)


