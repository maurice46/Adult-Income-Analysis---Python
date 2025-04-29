import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('adult.data', header=None, na_values=' ?')

# Check for duplicates and remove them
data.drop_duplicates(inplace=True)

# Define column names
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# Drop rows with missing values
data.dropna(inplace=True)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 
                                      'occupation', 'relationship', 'race', 
                                      'sex', 'native-country'], drop_first=True)

# Define features and target variable
X = data.drop(columns=['income'])  # Dropping target variable
Y = data['income'].map({' <=50K': 0, ' >50K': 1})  # Convert income to binary


# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, y_pred)
matrix = confusion_matrix(Y_test, y_pred)

# for better labeling
matrix = pd.DataFrame(matrix, 
                     index=["Actual: <=50K", "Actual: >50K"], 
                     columns=["Predicted: <=50K", "Predicted: >50K"])

print(f"Accuracy Score: {accuracy:.4f}")
print(matrix)

