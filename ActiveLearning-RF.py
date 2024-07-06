import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

# Assuming df_gc and df_nogc_filtered are already defined and contain the data

# Add labels to the dataframes
df_gc['label'] = 0
df_nogc_filtered['label'] = 1

# Combine the dataframes
df_all = pd.concat([df_gc, df_nogc_filtered])

# Separate features and labels
X = df_all.drop(columns=['label'])
y = df_all['label']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

# Undersample the majority class (Non-GCs) in the initial training set
undersample = RandomUnderSampler(sampling_strategy=0.1, random_state=42)  # Example: keep 10% of Non-GCs
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train, y_train)

# Oversample the minority class (GCs) in the initial training set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_resampled, y_train_resampled)

# Initialize and train the classifier on the entire resampled training set
classifier = RandomForestClassifier(n_estimators=100, class_weight={0: 10, 1: 1}, random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# Initialize the ActiveLearner with the pre-trained classifier
learner = ActiveLearner(
    estimator=classifier,
    X_training=X_train_resampled, y_training=y_train_resampled,
    query_strategy=uncertainty_sampling,
)

# Active learning loop
iterations = 10  # Define the number of active learning iterations
n_queries_per_iteration = 50  # Increase the number of instances to label per iteration

X_pool = X_train  # Use the original training set as the pool for querying
y_pool = y_train

for _ in range(iterations):
    # Query the instances from the pool to be labeled
    query_idx, query_instance = learner.query(X_pool)

    if len(query_idx) < n_queries_per_iteration:
        print(f"Not enough instances to sample for iteration {_ + 1}. Ending active learning.")
        break

    # Simulate or perform manual labeling (replace with actual labeling process)
    labeled_idx = np.random.choice(query_idx, size=n_queries_per_iteration, replace=False)
    X_label, y_label = X_pool[labeled_idx], y_pool[labeled_idx]

    # Teach the ActiveLearner with the newly labeled instances
    learner.teach(X=X_label, y=y_label)

    # Remove the newly labeled instances from the pool
    X_pool = np.delete(X_pool, labeled_idx, axis=0)
    y_pool = np.delete(y_pool, labeled_idx, axis=0)

    # Optionally, evaluate the model's performance after each iteration
    y_pred = learner.predict(X_test)
    print(f"Iteration {_ + 1} - Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Iteration {_ + 1} - Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Final evaluation
y_pred_final = learner.predict(X_test)
print("Final Evaluation - Classification Report:\n", classification_report(y_test, y_pred_final))
print("Final Evaluation - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_final))

# Apply the trained model to a new sample to find new GCs
X_sample_scaled = scaler.transform(X_sample)  # Assuming X_sample is your new sample data
y_pred_sample = learner.predict(X_sample_scaled)
y_prob_sample = learner.predict_proba(X_sample_scaled)

# Add the predictions and probabilities to the original sample dataframe
df_all_clean["Label"] = y_pred_sample
df_all_clean['Prob(GC)'] = y_prob_sample[:, 0]
df_all_clean['Prob(Non-GC)'] = y_prob_sample[:, 1]

# Save the GC classified instances to a new DataFrame and export to CSV
df_gc_only = df_all_clean[df_all_clean['Label'] == 0]
df_gc_only.to_csv('predicted_GC_results_only.csv', index=False)

# Count the number of objects with each label
label_counts = df_all_clean['Label'].value_counts()
print("Count of each label:\n", label_counts)
