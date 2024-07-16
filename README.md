# Eye Cataract Detection

## Overview

This project focuses on detecting eye cataracts using a convolutional neural network (CNN) model, specifically the VGG19 model. For web deployment, the Python library Streamlit is used.

## Key Steps

### Data Preparation

1. **Collect and Prepare Dataset**: 
   - Gather eye images labeled with cataract and non-cataract categories.
   - Ensure a balanced distribution of the dataset.

### Model Architecture

2. **Load VGG19 Model**: 
   - Use the pre-trained VGG19 model, excluding its top layer.
   
3. **Freeze Layers**: 
   - Prevent the pre-trained layers from being updated during training.
   
4. **Add Custom Layers**: 
   - Incorporate additional layers specific to cataract detection, such as Flatten, Dropout, and Dense layers.

### Model Compilation

5. **Optimizer and Loss Function**: 
   - Choose an optimizer like Adam.
   - Use binary cross-entropy as the loss function.
   
6. **Evaluation Metrics**: 
   - Set accuracy as the primary evaluation metric.

### Model Training

7. **Augmented Data**: 
   - Train the model using the augmented dataset.
   
8. **Callbacks**: 
   - Implement `ModelCheckpoint` to save the best model.
   - Use `EarlyStopping` to halt training if performance stops improving.

### Model Evaluation

9. **Performance Metrics**: 
   - Evaluate the model on a separate test set to measure its accuracy and loss.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/yourrepository.git](https://github.com/Naman-bangari/Eye-cataract-detection.git
    ```

2. **Install the Required Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

