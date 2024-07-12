import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import tensorflow as tf
from ultralytics import YOLO


model_path = 'yolov8s.pt'
model = YOLO(model_path)


model_architecture_path = 'model2.json'
model_weights_path = 'gender_classification_model.h5'


with open(model_architecture_path, 'r') as json_file:
    loaded_model_json = json_file.read()
    gender_model = tf.keras.models.model_from_json(loaded_model_json)


gender_model.load_weights(model_weights_path)

 
def classify_gender(image):
    image = cv2.resize(image, (96, 96))  
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  

    gender_prediction = gender_model.predict(image)
    return gender_prediction


def detect_car_color(car_img):
    car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB)
    pixels = car_img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=8)  
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)

    return dominant_color


def get_color_label(dominant_color):
    color_labels = {
        'Red': [255, 0, 0],
        'Green': [0, 255, 0],
        'Blue': [0, 0, 255],
        'Yellow': [255, 255, 0],
        'White': [255, 255, 255],
        'Black': [0, 0, 0],
        'Silver': [192, 192, 192],
        'Maroon': [128, 0, 0],
        'Olive': [128, 128, 0],
        'Purple': [128, 0, 128],
        'Teal': [0, 128, 128],
        'Navy': [0, 0, 128],
        'Orange': [255, 165, 0],
        'Pink': [255, 192, 203],
        'Brown': [165, 42, 42]
    }

    gray_threshold = 50  
    distances = {color: np.linalg.norm(dominant_color - np.array(rgb)) for color, rgb in color_labels.items()}
    color_label = min(distances, key=distances.get)

    if np.linalg.norm(dominant_color - np.array([128, 128, 128])) < gray_threshold:
        color_label = 'Unknown'  

    return color_label


st.title("Traffic Image Analysis")
st.write("Detect cars, swap car colors, and classify gender of detected people in the uploaded image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    height, width, _ = image.shape

    
    results = model(image)

    
    car_count = 0
    male_count = 0
    female_count = 0
    car_colors = []

    
    for result in results[0].boxes:
        cls = int(result.cls[0])
        x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())

        if cls == 0:  
            person_img = image[y1:y2, x1:x2]
            gender_prediction = classify_gender(person_img)

            if gender_prediction[0][0] > 0.5:
                male_count += 1
                label = 'Male'
            else:
                female_count += 1
                label = 'Female'
            color = (0, 255, 0)  

        elif cls == 2:  
            car_count += 1
            car_img = image[y1:y2, x1:x2]
            dominant_color = detect_car_color(car_img)
            color_label = get_color_label(dominant_color)
            car_colors.append(color_label)
            label = f'Car ({color_label})'
            color = (255, 0, 0) 

            
            if color_label == 'Red':
                color_label = 'Blue'
            elif color_label == 'Blue':
                color_label = 'Red'
            
            
            if color_label == 'Red':
                car_img[:, :, 0] = 255
                car_img[:, :, 1] = 0
                car_img[:, :, 2] = 0
            elif color_label == 'Blue':
                car_img[:, :, 0] = 0
                car_img[:, :, 1] = 0
                car_img[:, :, 2] = 255
            
            image[y1:y2, x1:x2] = car_img

        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    
    st.image(image, caption='Processed Image', use_column_width=True)
    st.write(f"Number of Cars: {car_count}")
    st.write(f"Number of Males: {male_count}")
    st.write(f"Number of Females: {female_count}")
    st.write(f"Car Colors: {', '.join([color for color in car_colors if color != 'Unknown'])}")
