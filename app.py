import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = pd.read_csv('approved_data.csv')

# Title of the Streamlit app
st.title("Life Insurance Approval Prediction")

# Create tabs
tabs = st.tabs(["EDA", "Modeling", "Scoring"])

with tabs[0]:
    st.header("Exploratory Data Analysis (EDA)")
    
    # Display the dataset
    st.write("Dataset Overview:")
    st.write(data.head())
    
    # Show basic information about the dataset
    st.write("Basic Information:")
    st.write(data.info())
    
    # Show summary statistics
    st.write("Summary Statistics:")
    st.write(data.describe())
    
    # Show the distribution of the target variable
    st.write("Distribution of Approved Target:")
    st.bar_chart(data['Approved'].value_counts())
    
    # Check for missing values
    st.write("Missing Values:")
    st.write(data.isnull().sum())

    # Correlation matrix
    st.write("Correlation Matrix:")
    st.write(data.corr())
    st.write("Correlation Heatmap:")
    st.heatmap(data.corr(), annot=True, cmap='coolwarm')

with tabs[1]:
    st.header("Modeling")

    # Separating features and target variable
    X = data.drop(columns=['Customer ID', 'Approved'])  # Ensure Customer ID is not included
    y = data['Approved']

    # Identify categorical and numerical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ]
    )

    # Define the models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to evaluate models
    def evaluate_models(models, X_train, y_train, X_test, y_test):
        results = {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
        return results

    # Evaluate the models
    model_accuracies = evaluate_models(models, X_train, y_train, X_test, y_test)

    # Display results in a table format
    results_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
    st.write("Model Accuracy Comparison:")
    st.table(results_df)

    # Show a bar chart of the results
    st.bar_chart(results_df)

with tabs[2]:
    st.header("Scoring")

    st.write("This section could include additional scoring metrics, model performance on validation data, or even a tool to allow users to input new data and receive predictions.")
    # Example of scoring logic (add actual implementation as needed)
    selected_model_name = st.selectbox("Select Model for Scoring", list(models.keys()))
    selected_model = models[selected_model_name]
    
    # Fit the selected model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', selected_model)])
    pipeline.fit(X_train, y_train)
    
    # Score the model on the test set
    score = pipeline.score(X_test, y_test)
    st.write(f"Accuracy of {selected_model_name} on test data: {score:.2f}")
    
    # Add an option for user input and prediction (optional)
    # This could include a form or widgets to collect input and predict approval status
