import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
df = pd.read_csv('datasets/wbc9.csv')

# Extract features and labels (case-insensitive for 'class' or 'label')
label_col = next(col for col in df.columns if col.lower() in ['class', 'label'])
X = df.drop(columns=[label_col]).values

# Automatically detect class names and assign integer labels
class_names = df[label_col].unique()
class_to_int = {name: idx for idx, name in enumerate(class_names)}
y = df['class'].map(class_to_int).values
print()

def train_and_evaluate(X, y, class_names, description):
    # Use all data for training (no split)
    X_train = X
    y_train = y

    # Train SVM with linear kernel on all data
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Build the hyperplane equations as strings for multiclass
    hyperplanes = []
    normalized_hyperplanes = []
    for class_idx, (coefs, intercept) in enumerate(zip(clf.coef_, clf.intercept_)):
        # Build original hyperplane equation
        original_terms = []
        for idx, coef in enumerate(coefs):
            sign = '+' if coef >= 0 and idx > 0 else ''
            original_terms.append(f"{sign}{coef:.4f}*{df.columns[idx]}")
        original_hyperplane = f"y_{class_idx} = " + " ".join(original_terms)
        if intercept >= 0:
            original_hyperplane += f" + {intercept:.4f}"
        else:
            original_hyperplane += f" - {abs(intercept):.4f}"
        hyperplanes.append(original_hyperplane)
        
        # Check if any coefficients are negative
        has_negative = np.any(coefs < 0)
        
        # Normalize coefficients based on whether they contain negative values
        if has_negative:
            # Normalize to [-1, 1] range
            coef_min, coef_max = np.min(coefs), np.max(coefs)
            if coef_max != coef_min:  # Avoid division by zero
                normalized_coefs = 2 * ((coefs - coef_min) / (coef_max - coef_min)) - 1
            else:
                normalized_coefs = coefs
        else:
            # Normalize to [0, 1] range
            coef_min, coef_max = np.min(coefs), np.max(coefs)
            if coef_max != coef_min:  # Avoid division by zero
                normalized_coefs = (coefs - coef_min) / (coef_max - coef_min)
            else:
                normalized_coefs = coefs
        
        # Build normalized hyperplane equation
        normalized_terms = []
        for idx, norm_coef in enumerate(normalized_coefs):
            sign = '+' if norm_coef >= 0 and idx > 0 else ''
            normalized_terms.append(f"{sign}{norm_coef:.4f}*{df.columns[idx]}")
        normalized_hyperplane = f"y_{class_idx} = " + " ".join(normalized_terms)
        if intercept >= 0:
            normalized_hyperplane += f" + {intercept:.4f}"
        else:
            normalized_hyperplane += f" - {abs(intercept):.4f}"
        normalized_hyperplanes.append(normalized_hyperplane)

    print(f"Results for {description}:")
    # Print the original and normalized hyperplanes
    for i, (hyperplane, normalized_hyperplane) in enumerate(zip(hyperplanes, normalized_hyperplanes)):
        print(f"Original hyperplane for {class_names[i]}:", hyperplane)
        print(f"Normalized hyperplane for {class_names[i]}:", normalized_hyperplane)
        print()

    # Use the hyperplanes to classify all data
    X_test = X
    y_test = y
    y_test_pred = clf.predict(X_test)

    # Print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    print("confusion matrix:\n", conf_matrix_df)

    # Print the accuracy
    print("accuracy:", accuracy_score(y_test, y_test_pred), "=", str(round(accuracy_score(y_test, y_test_pred) * 100, 2)) + "%\n")

# Test with original (not normalized) data
train_and_evaluate(X, y, class_names, "original (not normalized) data")

# Test with min-max normalized data
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
train_and_evaluate(X_norm, y, class_names, "min-max normalized data")

# Test with z-score normalized data
zscaler = StandardScaler()
X_zscore = zscaler.fit_transform(X)
train_and_evaluate(X_zscore, y, class_names, "z-score normalized data")
