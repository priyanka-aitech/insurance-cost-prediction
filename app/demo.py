import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the model and LabelEncoder
model = joblib.load('insurance-price-prediction/random_forest_model.pkl')
le_combined = joblib.load('insurance-price-prediction/le_combined.pkl')

# Function for the demo
def demo():
    st.title("Medical Insurance Price Prediction")

    # Sidebar for user input
    st.sidebar.header("Input Parameters")

    def user_input_features():
        age = st.sidebar.number_input("Age", min_value=18, max_value=120, value=30)
        sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
        bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=1)
        smoker = st.sidebar.selectbox("Smoker", options=["Yes", "No"])
        region = st.sidebar.selectbox("Region", options=["Northwest", "Northeast", "Southwest", "Southeast"])

        # Encode categorical columns
        sex_encoded = le_combined.transform([sex])[0]
        smoker_encoded = le_combined.transform([smoker])[0]
        region_encoded = le_combined.transform([region])[0]

        # Prepare input data as DataFrame
        data = {
            'age': age,
            'sex': sex_encoded,
            'bmi': bmi,
            'children': children,
            'smoker': smoker_encoded,
            'region': region_encoded
        }
        return pd.DataFrame([data])

    # Get user input
    input_df = user_input_features()

    # Prediction Button
    if st.button("Predict Insurance Price"):
        # Make prediction
        prediction = model.predict(input_df)
        predicted_price = prediction[0]

        # Display results with detailed explanation
        st.markdown(
            f"""
            <div style='background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-top: 20px; text-align: center;'>
                <h3 style='color: #4CAF50;'>Prediction Result</h3>
                <p><b>Predicted Medical Insurance Price:</b> ${predicted_price:.2f}</p>
                <p>This prediction considers multiple factors, such as:</p>
                <ul style='text-align: left;'>
                    <li><b>Age</b>: Older individuals may have higher insurance costs.</li>
                    <li><b>Sex</b>: Gender can impact premium rates based on statistical risk.</li>
                    <li><b>BMI</b>: Higher BMI may increase premiums due to associated health risks.</li>
                    <li><b>Number of Children</b>: Having more dependents can influence the insurance price.</li>
                    <li><b>Smoker Status</b>: Smokers usually have higher premiums due to increased health risks.</li>
                    <li><b>Region</b>: Geographic region may affect insurance rates due to healthcare cost variations.</li>
                </ul>
                <p>These inputs help the model estimate the appropriate insurance price for the individual, 
                ensuring accurate premium calculation and better financial planning.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
# Function for the documentation
def documentation():
    st.title("Documentation: Medical Insurance Price Prediction App")

    st.write(
        """
        ## Project Description
        The **Medical Insurance Price Prediction App** is a Streamlit-based application designed to predict medical insurance prices based on user input. 
        The app utilizes a pre-trained machine learning model to estimate insurance costs, providing users with an interactive platform to input their details and visualize the predicted price. 

        ## Usage Scenarios
        1. **Insurance Price Estimation**: Users can input personal and medical details to receive an estimate of their potential medical insurance cost.
        2. **Data Visualization**: The app provides various charts to help users understand how their input features impact the prediction and compare the predicted price against a baseline or historical distribution.
        """
    )

    st.header("1. Data Collection")
    st.write(
        """
        A dataset containing features such as age, sex, BMI, number of children, smoking status, and region, along with the target variable (insurance price), was collected from public sources or health insurance records. 
        This data is essential for training the machine learning model to predict insurance prices based on user inputs.
        """
    )

    st.header("2. Data Uploading")
    st.code("""
    import pandas as pd

    # Load the dataset
    data = pd.read_csv('insurance.csv')

    # Display the first few rows
    data.head()
    """, language='python')

    st.write(
        """
        **Purpose**: Load the dataset into a Pandas DataFrame for preprocessing.

        **Details**: The `pd.read_csv` function reads the CSV file containing the insurance data, allowing us to examine its structure.
        """
    )

    st.header("3. Basic Exploration of the Dataset")
    st.code("""
    # Display the shape of the dataset
    data_shape = data.shape

    # Display basic statistics
    data_description = data.describe()

    # Check for missing values
    missing_values = data.isnull().sum()

    data_shape, data_description, missing_values
    """, language='python')

    st.write(
        """
        **Purpose**: Gain initial insights into the dataset.

        **Details**:
        - **Shape**: The `data.shape` function provides the number of rows and columns, giving an overview of the dataset size.
        - **Statistics**: The `data.describe()` function returns basic statistics (count, mean, min, max, quartiles) for numeric features, helping us understand their distributions.
        - **Missing Values**: The `data.isnull().sum()` function checks for any missing values in the dataset, which is critical for determining the need for data cleaning and preprocessing.
        """
    )

    st.header("4. Exploratory Data Analysis (EDA)")
    st.code("""
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Visualize the distribution of the target variable
    sns.histplot(data['charges'], kde=True)
    plt.title('Distribution of Insurance Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.show()
    """, language='python')

    st.write(
        """
        **Purpose**: Understand the dataset better through visualizations.

        **Details**: EDA helps to identify trends, patterns, and anomalies in the data, guiding the preprocessing and modeling steps.
        """
    )

    st.header("5. Feature Scaling")
    st.code("""
    from sklearn.preprocessing import StandardScaler

    # Scaling numeric features
    scaler = StandardScaler()
    data[['age', 'bmi', 'children']] = scaler.fit_transform(data[['age', 'bmi', 'children']])
    """, language='python')

    st.write(
        """
        **Purpose**: Normalize numeric features to improve model performance.

        **Details**: Scaling ensures that all features contribute equally to the distance calculations in models like regression or k-nearest neighbors.
        """
    )

    st.header("6. Train-Test Split")
    st.code("""
    from sklearn.model_selection import train_test_split

    # Splitting the dataset into training and testing sets
    X = data.drop('charges', axis=1)
    y = data['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """, language='python')

    st.write(
        """
        **Purpose**: Split the dataset into training and testing sets to evaluate model performance.

        **Details**: This step ensures that the model is tested on unseen data, helping to prevent overfitting.
        """
    )

    st.header("7. Model Initialization")
    st.code("""
    from sklearn.ensemble import RandomForestRegressor

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    """, language='python')

    st.write(
        """
        **Purpose**: Prepare the machine learning model for training.

        **Details**: A Random Forest Regressor is chosen for its robustness and ability to capture complex relationships.
        """
    )

    st.header("8. Model Building")
    st.code("""
    # Train the model
    model.fit(X_train, y_train)
    """, language='python')

    st.write(
        """
        **Purpose**: Fit the model to the training data.

        **Details**: The model learns the relationships between the features and the target variable (insurance charges) during this phase.
        """
    )

    st.header("9. Model Evaluation")
    st.code("""
    from sklearn.metrics import mean_squared_error, r2_score

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')
    """, language='python')

    st.write(
        """
        **Purpose**: Assess the model's performance.

        **Details**: Metrics such as Mean Squared Error (MSE) and R² Score provide insights into how well the model predicts unseen data.
        """
    )

    st.header("10. Saving the Model")
    st.code("""
    import joblib

    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(le_combined, 'le_combined.pkl')
    """, language='python')

    st.write(
        """
        **Purpose**: Persist the trained model for future use.

        **Details**: The model and any necessary encoders are saved to files, allowing the app to load them for predictions without retraining.
        """
    )

    st.header("11. Deploying the Model in Streamlit")
    st.code("""
    import streamlit as st

    # Load the saved model
    model = joblib.load('random_forest_model.pkl')

    # Streamlit app code to take user inputs and predict
    def user_input_features():
        # User input logic here
        pass

    # Make predictions and display results
    input_data = user_input_features()
    predicted_price = model.predict(input_data)
    st.write(f'Predicted Insurance Price: ${predicted_price[0]:.2f}')
    """, language='python')

    st.write(
        """
        **Purpose**: Integrate the model into a Streamlit app for user interaction.

        **Details**: The app collects user inputs, invokes the model for predictions, and displays results in real-time.
        """
    )

    st.header("12. Final Results")
    st.write(
        """
        After deploying the model, users can input their details, and the app will provide a predicted medical insurance price based on the trained model. 
        Various visualizations and insights can also be displayed to enhance user understanding of the prediction process.
        """
    )

    st.header("Conclusion")
    st.write(
        """
        This documentation provides a comprehensive overview of the Medical Insurance Price Prediction App. It details each step, including data collection, preprocessing, model building, evaluation, and deployment. 
        Users can easily input their details to obtain real-time predictions, making the app a valuable tool for estimating medical insurance costs.
        """
    )

# Main app logic
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Demo", "Documentation"])

if selection == "Demo":
    demo()
elif selection == "Documentation":
    documentation()
