import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
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

    # Train SVM with linear kernel on all data, forcing intercept to 0 with fit_intercept=False
    clf = LinearSVC(fit_intercept=False)
    clf.fit(X_train, y_train)

    # Build the hyperplane equations as strings for multiclass
    hyperplanes = []
    normalized_hyperplanes = []
    angle_hyperplanes = []
    for class_idx, coefs in enumerate(clf.coef_):
        intercept = clf.intercept_[class_idx] if hasattr(clf.intercept_, '__len__') else clf.intercept_
        # Build original hyperplane equation
        original_terms = []
        for idx, coef in enumerate(coefs):
            sign = '+' if coef >= 0 and idx > 0 else ''
            original_terms.append(f"{sign}{coef:.4f}*{df.columns[idx]}")
        original_hyperplane = f"y_{class_idx} = " + " ".join(original_terms)
        # Intercept is forced to 0 by fit_intercept=False
        original_hyperplane += f" + {intercept:.4f}" if intercept else " + 0.0000"
        hyperplanes.append(original_hyperplane)
        
        # L2 normalization: scale coefficients to unit length
        coef_norm = np.linalg.norm(coefs)
        if coef_norm != 0:  # Avoid division by zero
            normalized_coefs = coefs / coef_norm
        else:
            normalized_coefs = coefs
        
        # Build normalized hyperplane equation
        normalized_terms = []
        for idx, norm_coef in enumerate(normalized_coefs):
            sign = '+' if norm_coef >= 0 and idx > 0 else ''
            normalized_terms.append(f"{sign}{norm_coef:.4f}*{df.columns[idx]}")
        normalized_hyperplane = f"y_{class_idx} = " + " ".join(normalized_terms)
        # Intercept is forced to 0 by fit_intercept=False
        normalized_hyperplane += f" + {intercept:.4f}" if intercept else " + 0.0000"
        normalized_hyperplanes.append(normalized_hyperplane)
        
        # Convert L2-normalized coefficients to angles (in degrees)
        # L2-normalized components are already cosines to each axis
        angles_degrees = []
        for norm_coef in normalized_coefs:
            # Clamp coefficient to [-1, 1] range for arccos
            clamped_coef = np.clip(norm_coef, -1, 1)
            # Convert cosine to angle in radians, then to degrees
            angle_rad = np.arccos(clamped_coef)
            angle_deg = np.degrees(angle_rad)
            angles_degrees.append(angle_deg)
        
        # Build angle-based hyperplane equation
        angle_terms = []
        for idx, angle_deg in enumerate(angles_degrees):
            sign = '+' if angle_deg <= 90 and idx > 0 else ''  # <= 90 degrees means positive direction
            angle_terms.append(f"{sign}{angle_deg:.2f}°*{df.columns[idx]}")
        angle_hyperplane = f"y_{class_idx} = " + " ".join(angle_terms)
        # Intercept is forced to 0 by fit_intercept=False
        angle_hyperplane += f" + {intercept:.4f}" if intercept else " + 0.0000"
        angle_hyperplanes.append(angle_hyperplane)

    print(f"Results for {description}:")
    # Print the original, normalized, and angle-based hyperplanes
    for i, (hyperplane, normalized_hyperplane, angle_hyperplane) in enumerate(zip(hyperplanes, normalized_hyperplanes, angle_hyperplanes)):
        print(f"Original hyperplane for {class_names[i]}:", hyperplane)
        print(f"Normalized hyperplane for {class_names[i]}:", normalized_hyperplane)
        print(f"Angle-based hyperplane for {class_names[i]}:", angle_hyperplane)
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
