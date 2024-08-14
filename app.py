import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = pd.read_csv('approved_data.csv')
st.write("Dataset loaded successfully!")

# Function to evaluate models
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name in models.keys():  # Create a list of model names to iterate over
        model = models[name]
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results

# Streamlit interface
st.title('Life Insurance Underwriting')

# Create tabs
tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "Scoring"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")

    # Display basic dataset information
    st.subheader("Basic Information")
    st.write("Dataset shape:", data.shape)
    st.write("Dataset columns:", data.columns.tolist())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Histograms for categorical features
    st.subheader("Histograms for Smoking Status, Medical History, and Alcohol Consumption")

    categorical_columns = ['Smoking Status', 'Medical History', 'Alcohol Consumption']

    for column in categorical_columns:
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=False, bins=10, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        
        # Set custom labels for Smoking Status
        if column == 'Smoking Status':
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels(['Non-Smoker (0.0)', 'Smoker (1.0)'])
        
        st.pyplot(fig)

    # Target distribution
    st.subheader("Target Distribution")
    
    # Calculate approval rate
    approval_rate = data['Approved'].mean() * 100
    st.write(f"Approval Rate: {approval_rate:.2f}%")
    
    fig, ax = plt.subplots()
    sns.countplot(x='Approved', data=data, ax=ax)
    ax.set_xlabel("Approved")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Pairplot for numerical features
    st.subheader("Pairplot")
    if len(data.select_dtypes(include=['int64', 'float64']).columns) > 1:
        fig = sns.pairplot(data, hue='Approved')
        st.pyplot(fig)
    else:
        st.write("Not enough numerical features for pairplot.")

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Heatmap of correlations
    st.subheader("Heatmap of Correlations")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("Modeling")

    # Separating features and target variable
    X = data.drop(columns=['Approved'])
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

    # Evaluate the models
    model_accuracies = evaluate_models(models, X_train, y_train, X_test, y_test)

    # Display results
    st.subheader("Model Accuracies")
    results_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
    st.write(results_df)

with tab3:
    st.header("Scoring")

    st.write("This tab could be used for additional scoring metrics or detailed analysis.")
