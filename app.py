+import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prince  # For Correspondence Analysis
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
    for name in models.keys():
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

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Histograms for categorical features with percentages
    st.subheader("Histograms for Smoking Status, Medical History, and Alcohol Consumption")

    categorical_columns = ['Smoking Status', 'Medical History', 'Alcohol Consumption']

    for column in categorical_columns:
        fig, ax = plt.subplots()
        counts = data[column].value_counts(normalize=True) * 100  # Calculate percentage
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel("Percentage")

        # Annotate bars with percentages
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                        textcoords='offset points')

        # Set custom x-axis labels for Smoking Status
        if column == 'Smoking Status':
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Non-Smoker (0)', 'Smoker (1)'])

        # Set custom x-axis labels for Medical History
        if column == 'Medical History':
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['No Disease (0)', 'Diabetes (1)', 'Hypertension (2)', 'Heart Disease (3)'])

        # Set custom x-axis labels for Alcohol Consumption
        if column == 'Alcohol Consumption':
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(['Never (0)', 'Low (1)', 'Moderate (2)', 'High (3)'])

        st.pyplot(fig)

    # Correspondence Analysis
    st.subheader("Correspondence Analysis")

    ca_columns = [
        'Gender', 'Smoking Status', 'Medical History', 'Occupation',
        'Term Length', 'Family History', 'Physical Activity Level',
        'Alcohol Consumption', 'Premium Payment Frequency'
    ]

    # Filter out missing columns
    ca_columns = [col for col in ca_columns if col in data.columns]

    if len(ca_columns) > 1:
        ca_data = data[ca_columns]

        # Perform Correspondence Analysis
        ca = prince.CA(n_components=2)
        ca = ca.fit(ca_data)

        # Get row and column coordinates
        row_coords = ca.row_coordinates(ca_data)
        col_coords = ca.column_coordinates(ca_data)

        # Plot the biplot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the row coordinates
        sns.scatterplot(x=row_coords[0], y=row_coords[1], hue=data['Approved'], ax=ax, marker='o', label='Rows')

        # Plot the column coordinates
        sns.scatterplot(x=col_coords[0], y=col_coords[1], ax=ax, marker='x', label='Columns', color='red')

        for i, col in enumerate(ca_columns):
            ax.text(col_coords[0][i], col_coords[1][i], col, color='red', fontsize=12)

        ax.set_title('Correspondence Analysis Biplot')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        st.pyplot(fig)
    else:
        st.write("Not enough variables available for Correspondence Analysis.")
        
# Target distribution
st.subheader("Target Distribution")

# Calculate approval and rejection counts
approval_count = data['Approved'].sum()
rejection_count = len(data) - approval_count

# Calculate approval rate
approval_rate = approval_count / len(data) * 100
rejection_rate = 100 - approval_rate

# Display counts and rates
st.write(f"Number of Accepted Applications: {approval_count}")
st.write(f"Number of Rejected Applications: {rejection_count}")
st.write(f"Approval Rate: {approval_rate:.2f}%")
st.write(f"Rejection Rate: {rejection_rate:.2f}%")

# Plot the distribution
fig, ax = plt.subplots()
sns.countplot(x='Approved', data=data, ax=ax)
ax.set_xlabel("Approved (1) / Rejected (0)")
ax.set_ylabel("Count")

# Annotate bars with counts
for p in ax.patches:
    count = f'{p.get_height()}'
    ax.annotate(count, (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
                textcoords='offset points')

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

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(model_accuracies, orient='index', columns=['Accuracy'])
    results_df = results_df.sort_values(by='Accuracy', ascending=False)

    # Display the accuracies
    st.subheader("Model Accuracies")
    st.write(results_df)

    # Highlight the best model dynamically
    best_model_name = results_df.index[0]
    best_model_accuracy = results_df.iloc[0, 0]
    st.subheader(f"Best Model: {best_model_name}")
    st.write(f"Accuracy: {best_model_accuracy:.2f}%")

    # Select and train the best model
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

with tab3:
    st.header("Scoring")

    # File uploader for custom data
    uploaded_file = st.file_uploader("Upload your dataset for scoring", type="csv")
    
    if uploaded_file is not None:
        # Load the uploaded data
        custom_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.write(custom_data.head())

        # Define approval criteria
        def approve_application(row):
            # Example criteria (adjust based on actual underwriting policies)
            if (18 <= row['Age'] <= 65 and
                row['Smoking Status'] == 0 and  # Assuming 0 is Non-Smoker
                18.5 <= row['BMI'] <= 24.9 and  # Healthy BMI range
                row['Medical History'] == 0 and  # Assuming 0 is No significant medical history
                row['Alcohol Consumption'] <= 2 and  # Assuming 2 or less is moderate/no consumption
                row['Family History of Disease'] == 0 and  # Assuming 0 is No significant family history
                row['Occupation'] != 'High Risk'):  # Assuming 'High Risk' occupation is identified
                return 1
            else:
                return 0

        # Apply criteria to create 'Approved' column
        custom_data['Approved'] = custom_data.apply(approve_application, axis=1)

        # Display the updated data with the 'Approved' column
        st.write("Scored data with 'Approved' column:")
        st.write(custom_data.head())

        # Calculate the approval rate
        approval_rate_calculated = custom_data['Approved'].mean() * 100
        st.write(f"Approval Rate: {approval_rate_calculated:.2f}%")

        # Count of approved and rejected forms
        approved_count = custom_data['Approved'].sum()
        rejected_count = len(custom_data) - approved_count
        st.write(f"Number of Approved Forms: {approved_count}")
        st.write(f"Number of Rejected Forms: {rejected_count}")

        # Download the scored data
        csv = custom_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scored Data",
            data=csv,
            file_name='scored_data.csv',
            mime='text/csv',
        )
