import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# App title
st.title("Data Science Learning App")

# Sidebar for navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Introduction", "Data Visualization", "Machine Learning"])

# 1. Introduction Section
if menu == "Introduction":
    st.header("Welcome to the Data Science Learning App!")
    st.write("""
        In this app, you will learn about fundamental data science concepts, including:
        - **Data Visualization**: Learn how to visualize and explore data.
        - **Machine Learning**: Build simple machine learning models.
        
        Explore the sections from the sidebar to get started!
    """)

# 2. Data Visualization Section
elif menu == "Data Visualization":
    st.header("Data Visualization")
    st.write("""
        Data visualization helps in understanding the patterns, trends, and relationships within the data.
        Here, we use the famous `Iris` dataset to demonstrate visualizations.
    """)

    # Load Iris dataset
    df = sns.load_dataset('iris')

    # Display the dataset
    if st.checkbox("Show Dataset"):
        st.write(df)

    # Select feature for visualization
    st.subheader("Feature-wise Visualization")
    feature = st.selectbox("Select a feature for visualization:", df.columns[:-1])

    # Plot histogram for the selected feature
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    st.pyplot(fig)

    # Plot pairplot for the dataset
    if st.checkbox("Show Pairplot (Relationships)"):
        st.subheader("Pairplot (Relationship between features)")
        fig = sns.pairplot(df, hue='species')
        st.pyplot(fig)

# 3. Machine Learning Section
elif menu == "Machine Learning":
    st.header("Basic Machine Learning")
    st.write("""
        In this section, you will learn how to build a basic machine learning model.
        We'll implement a **Linear Regression** model on the **Iris** dataset to predict petal width based on petal length.
    """)

    # Load Iris dataset
    df = sns.load_dataset('iris')

    # Select features for prediction
    X = df[['petal_length']]
    y = df['petal_width']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Show results
    st.subheader("Model Results")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    st.write(f"Model Coefficients: {model.coef_[0]}")
    st.write(f"Model Intercept: {model.intercept_}")

    # Plotting the prediction results
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='blue', label='Actual')
    ax.plot(X_test, y_pred, color='red', label='Predicted')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.legend()
    st.pyplot(fig)

    st.write("""
        This is a basic example of how linear regression works in machine learning. The red line shows the model's predictions based on the input feature (petal length).
    """)

