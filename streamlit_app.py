import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import tempfile
import pandas as pd
import os
import base64

st.title("Skin Disease Detection")

# Roboflow API configuration
API_KEY = "U9N0SYyfFJ7R5uJn7kYX"
MODEL_ID = "skin_disease_ak"
VERSION = "1"
API_URL = f"https://serverless.roboflow.com/inference/{MODEL_ID}/{VERSION}"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tfile.write(uploaded_file.read())
    image_path = tfile.name
    tfile.close()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        st.error("Could not read the image. Please upload a valid image file.")
    else:
        # Display the uploaded image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to trigger inference
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Read and encode the image
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                    
                    # Encode image to base64
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Prepare the request payload
                    payload = {
                        "api_key": API_KEY,
                        "image": image_base64
                    }
                    
                    headers = {
                        "Content-Type": "application/json"
                    }
                    
                    # Send the request
                    response = requests.post(
                        API_URL,
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Show predictions
                        if 'predictions' in result and result['predictions']:
                            # Get the top prediction
                            top_prediction = max(result['predictions'], key=lambda x: x.get('confidence', 0))
                            st.success(f"**Top Prediction:** {top_prediction['class']} (Confidence: {top_prediction['confidence']:.2f})")
                            
                            # Show all predictions in a table
                            pred_df = pd.DataFrame(result['predictions'])
                            st.subheader("All Predictions")
                            st.dataframe(pred_df[['class', 'confidence']])
                            
                            # Draw predictions on the image
                            annotated_image = image.copy()
                            for prediction in result['predictions']:
                                if all(k in prediction for k in ('x', 'y', 'width', 'height')):
                                    # Object detection: draw bounding box
                                    x0 = int(prediction['x'] - prediction['width'] / 2)
                                    y0 = int(prediction['y'] - prediction['height'] / 2)
                                    x1 = int(prediction['x'] + prediction['width'] / 2)
                                    y1 = int(prediction['y'] + prediction['height'] / 2)
                                    label = prediction['class']
                                    conf = prediction['confidence']
                                    cv2.rectangle(annotated_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
                                    cv2.putText(annotated_image, f"{label} {conf:.2f}", (x0, y0 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    # Classification: overlay class and confidence at the top
                                    label = prediction['class']
                                    conf = prediction['confidence']
                                    cv2.putText(annotated_image, f"{label} {conf:.2f}", (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Display annotated image
                            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            st.image(annotated_image_rgb, caption="Analysis Result", use_column_width=True)
                        else:
                            st.info("No predictions found for this image.")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        st.info("This might be due to an invalid API key or model configuration.")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please check your internet connection and try again.")
        
        # Clean up temporary file
        os.unlink(image_path) 