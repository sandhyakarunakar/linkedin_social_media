
# Train and evaluate Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the generated dataset from CSV
df = pd.read_csv("liinkedin_dataset.csv")

# Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
categorical_columns = ["Username", "Profile Picture", "Bio", "Job Search and Recruitment",
                       "Company Pages", "Groups and Communities", "Networking Opportunities",
                       "Learning and Development", "Analytics and Insights", "Skill Assessments",
                       "Alumni Networking", "Status"]
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features and target variable
X = df.drop(columns=["Status"])
y = df["Status"]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

print("Random Forest Accuracy:", rf_accuracy*100)


#OUTPUT:-

# Random Forest Accuracy: 73.1