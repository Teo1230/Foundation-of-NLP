import joblib
from utils import *
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
def weighted_accuracy(y_true, y_pred):
    total = 0
    for true, pred in zip(y_true, y_pred):
        distance = abs(int(true) - int(pred))
        partial_quotient = 1 / (2 ** distance)
        total += partial_quotient
    return total / len(y_true)


# Step 6: Train, evaluate, and save models with hyperparameter tuning
for model_name, (model, params) in models_and_params.items():
    print(f"Training {model_name} with GridSearchCV...")
    pipeline = make_pipeline(vectorizer, model)
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    # Evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    w_acc = weighted_accuracy(y_test, y_pred)

    print(f"{model_name} Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} WAccuracy: {w_acc:.4f}\n")

    # Save the model
    model_filename = f"{model_name.replace(' ', '_')}_acc_{accuracy}best_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"{model_name} saved as {model_filename}\n")

