import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from mtcnn import MTCNN
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Path to the static dataset CSV
dataset_file_path = 'C:/Users/DELL/Desktop/AIML/accessments/dataset_for_model_training_final.csv'

# Load and display the dataset
dataset_df = pd.read_csv(dataset_file_path)

# Load the pre-trained face classification model
model = tf.keras.models.load_model('face_detection_model_final_1.keras')

# Initialize the MTCNN face detector
detector = MTCNN()

# Streamlit app layout
st.title("Face Detection and Dataset Analysis App")

# Sidebar options
menu = st.sidebar.selectbox(
    "Menu",
    options=["Face Detection", "Model Metrics", "EDA & Visualizations"]
)

# Model metrics section
if menu == "Model Metrics":
    # Title of the app
    # st.title("Dataset and Model Metrics")
    # st.subheader("Dataset Preview")
    # st.dataframe(dataset_df)

    # Hard-coded model metrics data
    metrics_data = {
        'Metric': ['Validation Loss', 'Validation Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [0.1629, 0.9680, 0.9687, 1.0000, 0.9841]
    }

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(metrics_data)

    # Show Model Performance Metrics
    st.write("Model Performance Metrics:")
    # Show Model Performance Metrics
    st.subheader("Model Performance Metrics:")
    st.table(metrics_df)  # Using st.table() instead of st.dataframe()

    # Confusion Matrix Data
    confusion_matrix = [[0, 21], [0, 649]]
    st.write("Confusion Matrix:")
    st.write(confusion_matrix)

    # Plot metrics (e.g., accuracy and loss)
    # Hard-code accuracy and loss values for plotting (since you don't have them in a CSV here)
    accuracy_values = [0.9680]
    loss_values = [0.1480]

    # Create a plot for accuracy and loss
    fig = go.Figure()

    # Add accuracy trace
    fig.add_trace(go.Scatter(x=[0], y=accuracy_values, mode='lines+markers', name='Accuracy'))

    # Add loss trace
    fig.add_trace(go.Scatter(x=[0], y=loss_values, mode='lines+markers', name='Loss'))

    # Update layout with titles
    fig.update_layout(title="Model Accuracy and Loss", xaxis_title="Epoch", yaxis_title="Value")

    # Show the plot
    st.plotly_chart(fig)

# EDA section
elif menu == "EDA & Visualizations":
    st.header("Exploratory Data Analysis (EDA) and Visualizations")

    ## Display the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.dataframe(dataset_df.head())  # This will display the first 5 rows of the dataframe

    # Number of Faces per Image
    st.subheader("Number of Faces per Image")

    # Group by image_name and count the number of faces per image
    image_face_counts = dataset_df.groupby('image_name')['label'].count()

    # Create a figure
    plt.figure(figsize=(10, 5))

    # Plot the histogram with kde
    sns.histplot(image_face_counts, bins=30, kde=True)

    # Add title and labels
    plt.title('Distribution of Number of Faces per Image')
    plt.xlabel('Number of Faces per Image')
    plt.ylabel('Frequency')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Plotting the density (KDE plot) of the face_count
    st.subheader("Density Plot (KDE) of Face Count")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(dataset_df['face_count'], shade=True)
    plt.title("Density Plot of Face Count")
    plt.xlabel("Face Count")
    plt.ylabel("Density")
    st.pyplot()

    # Bounding Box Accuracy (Visualization)
    st.subheader("Bounding Box Accuracy Visualization")
    # Calculate the center of the bounding box (x_center, y_center)
    dataset_df['x_center'] = (dataset_df['x0'] + dataset_df['x1']) / 2
    dataset_df['y_center'] = (dataset_df['y0'] + dataset_df['y1']) / 2

    # Visualize bounding boxes on a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=dataset_df, x='x_center', y='y_center', hue='is_valid', palette="viridis")
    plt.title("Bounding Box Centers and Validity")
    plt.xlabel("X Center of Bounding Box")
    plt.ylabel("Y Center of Bounding Box")
    st.pyplot()

    # Distribution of Image Resolution
    st.subheader("Distribution of Image Resolutions")

    # Calculate the aspect ratio of the images (width / height)
    dataset_df['aspect_ratio'] = dataset_df['width'] / dataset_df['height']

    # Plot the distribution of image resolutions (width and height)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for width distribution
    sns.histplot(dataset_df['width'], bins=30, kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title("Distribution of Image Width")
    ax[0].set_xlabel("Width (Pixels)")
    ax[0].set_ylabel("Frequency")

    # Plot for height distribution
    sns.histplot(dataset_df['height'], bins=30, kde=True, ax=ax[1], color='salmon')
    ax[1].set_title("Distribution of Image Height")
    ax[1].set_xlabel("Height (Pixels)")
    ax[1].set_ylabel("Frequency")

    st.pyplot(fig)

    # Optional: Distribution of Aspect Ratio (width/height)
    st.subheader("Distribution of Aspect Ratios (Width/Height)")

    # Plot the aspect ratio distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(dataset_df['aspect_ratio'], bins=30, kde=True, color='green')
    plt.title("Distribution of Image Aspect Ratios (Width/Height)")
    plt.xlabel("Aspect Ratio (Width/Height)")
    plt.ylabel("Frequency")
    st.pyplot()

# Face Detection section
else:
    st.header("Face Detection with MTCNN")

    # Upload image input
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Open image using PIL
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        # Convert the image to a numpy array
        image = np.array(image.convert('RGB'))

        # Detect faces using MTCNN
        faces = detector.detect_faces(image)

        # If faces are detected, draw bounding boxes
        if len(faces) > 0:
            for face in faces:
                # Get bounding box coordinates
                x, y, width, height = face['box']
                # Draw rectangle around the face
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Convert image back to RGB for displaying in Streamlit
            image_with_bboxes = Image.fromarray(image)
            st.image(image_with_bboxes, caption='Face(s) Detected with Bounding Box', use_container_width=True)
        else:
            st.write("No face detected in the image.")

        # Prepare image for classification (resize and normalize as needed)
        image_resized = cv2.resize(image, (224, 224))  # Resize for model input
        image_input = np.expand_dims(image_resized, axis=0)  # Add batch dimension
        image_input = image_input / 255.0  # Normalize image

        # Predict using the face classification model
        prediction = model.predict(image_input)

        st.write("Prediction Score:", prediction[0])

        # Display the classification result
        if prediction > 0.98:  # Threshold for classification (you can adjust this)
            st.write("A face was detected and classified.")
        else:
            st.write("Not classified as a valid face.")
