import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('datasets/iris.csv')

# Extract features and labels
X = df.drop(columns=['class']).values

# Automatically detect class names and assign integer labels
class_names = df['class'].unique()
class_to_int = {name: idx for idx, name in enumerate(class_names)}
y = df['class'].map(class_to_int).values

def train_and_evaluate(X, y, class_names, description):
    # Use all data for training (no split)
    X_train = X
    y_train = y

    # Train SVM with linear kernel on all data
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Build the hyperplane equations as strings for multiclass
    hyperplanes = []
    for class_idx, (coefs, intercept) in enumerate(zip(clf.coef_, clf.intercept_)):
        terms = []
        for idx, coef in enumerate(coefs):
            sign = '+' if coef >= 0 and idx > 0 else ''
            terms.append(f"{sign}{coef:.4f}*{df.columns[idx]}")
        hyperplane = f"y_{class_idx} = " + " ".join(terms)
        if intercept >= 0:
            hyperplane += f" + {intercept:.4f}"
        else:
            hyperplane += f" - {abs(intercept):.4f}"
        hyperplanes.append(hyperplane)

    print(f"\nResults for {description}:")
    # Print the hyperplanes
    for hyperplane in hyperplanes:
        print(f"hyperplane for {class_names[hyperplanes.index(hyperplane)]}:", hyperplane)

    # Use the hyperplanes to classify all data
    X_test = X
    y_test = y
    y_test_pred = clf.predict(X_test)

    # Print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    print("confusion matrix:\n", conf_matrix_df)

    # Print the accuracy
    print("accuracy:", accuracy_score(y_test, y_test_pred))

# Test with original (not normalized) data
train_and_evaluate(X, y, class_names, "original (not normalized) data")

# Test with min-max normalized data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
train_and_evaluate(X_norm, y, class_names, "min-max normalized data")
