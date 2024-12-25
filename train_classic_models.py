from collections import Counter
import joblib
import os
from utils import *
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Step 1: Load and preprocess text data
txt = load_text()
st = sorted(txt)

texts, labels = [], []

for century in st:
    print(f"Century: {century} -> {len(txt[century])}")
    for data in txt[century]:
        text = prep(data)  # Preprocess the text data
        texts.append(text)
        labels.append(century)

# Step 2: Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 3: Augment classes to ensure at least 4 samples per class
label_counts = Counter(encoded_labels)
augmented_texts, augmented_labels = texts.copy(), list(encoded_labels)  # Convert encoded_labels to a list

for label, count in label_counts.items():
    if count < 4:  # Ensure at least 4 samples per class
        sample_indices = [i for i, lbl in enumerate(encoded_labels) if lbl == label]
        for _ in range(4 - count):  # Duplicate samples until the count reaches 4
            augmented_texts.append(texts[sample_indices[0]])  # Add the first sample for this label
            augmented_labels.append(label)

# Step 4: Train-test split with stratification
num_classes = len(label_encoder.classes_)
min_test_size = num_classes / len(augmented_labels)  # Ensure at least 1 sample per class in the test set

test_size = max(0.3, min_test_size)  # Adjust test_size to ensure no errors
print(f"Adjusted test size: {test_size:.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    augmented_texts,
    augmented_labels,
    test_size=test_size,
    random_state=42,
    stratify=augmented_labels
)

# Further split the training data into train and validation sets
val_size = max(0.2, num_classes / len(X_train))  # Ensure at least 1 sample per class in the validation set
print(f"Adjusted validation size: {val_size:.2f}")

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train,
    y_train,
    test_size=val_size,  # 20% or dynamically adjusted size
    random_state=42,
    stratify=y_train
)

# Step 5: TF-IDF Vectorizer with specified parameters
tfidf_vectorizer = TfidfVectorizer(
    min_df=1, max_features=None, strip_accents='unicode',
    analyzer='word', token_pattern=r'\b[^\d\W]+\b', ngram_range=(2, 4),
    use_idf=True, smooth_idf=True, sublinear_tf=True
)

# Step 6: Initialize models and their parameter grids for tuning
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
    }),
}

# Step 7: Define weighted accuracy function
def weighted_accuracy(y_true, y_pred):
    total = 0
    for true, pred in zip(y_true, y_pred):
        distance = abs(int(true) - int(pred))
        partial_quotient = 1 / (2 ** distance)
        total += partial_quotient
    return total / len(y_true)

# Step 8: Train, evaluate, and save models with hyperparameter tuning
min_class_size = min(Counter(y_train_final).values())  # Smallest class size in the training set
if min_class_size < 2:
    print("Using Leave-One-Out Cross-Validation (LOOCV) due to small class sizes")
    cv_strategy = LeaveOneOut()
else:
    n_splits = min(3, min_class_size)  # Use 3 splits or the minimum class size, whichever is smaller
    print(f"Using n_splits={n_splits} for GridSearchCV")
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for model_name, (model, params) in models_and_params.items():
    print(f"Training {model_name} with GridSearchCV...")
    pipeline = make_pipeline(tfidf_vectorizer, model)
    grid_search = GridSearchCV(pipeline, param_grid=params, cv=cv_strategy, scoring='accuracy', n_jobs=-1)

    # Train on X_train_final and validate on X_val
    grid_search.fit(X_train_final, y_train_final)

    # Validation evaluation
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_w_acc = weighted_accuracy(y_val, y_val_pred)

    print(f"{model_name} Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} Validation Accuracy: {val_accuracy:.4f}")
    print(f"{model_name} Validation WAccuracy: {val_w_acc:.4f}\n")

    # Test evaluation
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_w_acc = weighted_accuracy(y_test, y_test_pred)

    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    print(f"{model_name} Test WAccuracy: {test_w_acc:.4f}\n")

    # Save the model
    model_filename = f"{model_name.replace(' ', '_')}_val_acc_{val_accuracy:.4f}_test_acc_{test_accuracy:.4f}_best_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"{model_name} saved as {model_filename}\n")
