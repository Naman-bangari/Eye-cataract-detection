import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  
    return image

def predict_cataract(model, image):
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction[0][0]

def main():
    st.set_page_config(page_title="Cataract Detection App", page_icon=":eye:", layout="wide")
    
    
    st.markdown(
        """
        <style>
        
        .main {
            background-color: white !important;
            color: black !important;
        }

         h1,h3,h4,h5,h2,p {
            color: black !important;
        }
         .stButton button {
            color: black !important;
            background-color: #ffffff !important; 
            border-color: black !important; 
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title('Cataract Detection App')
    st.markdown("### Upload an image to predict if it has cataract")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((224, 224))
        st.image(resized_image, caption='Uploaded Image', use_column_width=False)

        model = load_model('vgg19.keras')
        
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = predict_cataract(model, image)
            
            st.subheader('Prediction Result')
            if prediction > 0.75:  
                st.success(f'This image is likely to have cataract .')
            else:
                st.info(f'This image is likely to be normal .')

            st.markdown("""
            ## What is Cataract?
            A cataract is a clouding of the lens in the eye, which leads to a decrease in vision. It is the most common cause of blindness and is conventionally treated with surgery.
            ### Symptoms
            - Blurry vision
            - Colors seem faded
            - Glare
            - Poor night vision
            - Double vision in one eye
            ### Risk Factors
            - Aging
            - Diabetes
            - Smoking
            - Prolonged exposure to sunlight
            """)
    st.image('1.jpg', caption='Example Image',width=300)
    
if __name__ == '__main__':
    main()
