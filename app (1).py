import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("Employee Salary Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    st.subheader("Select Target Column (what you want to predict):")
    target_col = st.selectbox("Choose target column", data.columns)

    if target_col:
        # Drop missing values
        data.dropna(inplace=True)

        # Encode categorical variables
        label_encoders = {}
        for column in data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

        # Split features and target
        X = data.drop(target_col, axis=1)
        y = data[target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model trained with accuracy: {acc:.2f}")
        st.subheader("Classification Report:")
        st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a CSV file to continue.")
