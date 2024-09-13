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

# 1. Friendly Introduction Section
if menu == "Introduction":
    st.header("Welcome to the Data Science Learning App!")
    st.write("""
        Hi there! This app is here to help you learn and explore key data science concepts in a fun, interactive way.

        Here's what you'll get to experience:
        - 📊 **Data Visualization**: See how visualizing data can help uncover hidden patterns and insights.
        - 🤖 **Machine Learning**: Get hands-on with a basic machine learning model using real-world data.

        Use the sidebar to explore the sections. Let's get started! 🚀
    """)

# 2. Data Visualization Section with Iris Photo
elif menu == "Data Visualization":
    st.header("Data Visualization and Fundamental Analysis")
    st.write("""
        Data visualization helps in understanding the patterns, trends, and relationships within the data.
        Here, we use the famous `Iris` dataset to demonstrate visualizations and basic data analysis.
    """)

    # Display an Iris flower photo
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
             caption="Iris Flower", use_column_width=True)

    # Load Iris dataset
    df = sns.load_dataset('iris')

    # Display the dataset
    if st.checkbox("Show Dataset"):
        st.write(df)

    # Basic statistics of the dataset
    if st.checkbox("Show Basic Statistics"):
        st.subheader("Basic Statistics")
        st.write(df.describe())

    # Select feature for visualization
    st.subheader("Feature-wise Visualization")
    feature = st.selectbox("Select a feature for visualization:", df.columns[:-1])

    # Plot histogram for the selected feature
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    st.pyplot(fig)

    # Additional visualizations: Boxplot and Scatter Plot
    st.subheader("Additional Visualizations")

    # 1. Boxplot to observe distributions and outliers
    st.write("**Boxplot**: Check distribution and outliers for the selected feature across species.")
    fig, ax = plt.subplots()
    sns.boxplot(x='species', y=feature, data=df, ax=ax)
    ax.set_title(f'Boxplot of {feature} by Species')
    st.pyplot(fig)

    # 2. Scatter Plot to see relationships between two variables
    st.write("**Scatter Plot**: Check the relationship between two numerical features.")
    feature_x = st.selectbox("Select X-axis feature:", df.columns[:-1], index=2)
    feature_y = st.selectbox("Select Y-axis feature:", df.columns[:-1], index=3)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df['species'], ax=ax)
    ax.set_title(f'Scatter Plot: {feature_x} vs {feature_y}')
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

# Footer: Developed by Miyoko Shimura with LinkedIn Link
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Developed by [Miyoko Shimura](https://www.linkedin.com/in/miyoko-shimura/)**  
""")
