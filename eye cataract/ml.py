import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def has_cataract(text):
    return 1 if "cataract" in text else 0

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
    df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
    left_cataract = df.loc[(df.C == 1) & (df.left_cataract == 1)]["Left-Fundus"].values
    right_cataract = df.loc[(df.C == 1) & (df.right_cataract == 1)]["Right-Fundus"].values
    left_normal = df.loc[(df.C == 0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250, random_state=42).values
    right_normal = df.loc[(df.C == 0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250, random_state=42).values
    cataract = np.concatenate((left_cataract, right_cataract), axis=0)
    normal = np.concatenate((left_normal, right_normal), axis=0)
    return cataract, normal

def create_dataset(image_category, label, dataset_dir, image_size):
    dataset = []
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir, img)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (image_size, image_size))
        except:
            continue
        dataset.append([np.array(image), np.array(label)])
    random.shuffle(dataset)
    return dataset

def build_model(input_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in vgg.layers:
        layer.trainable = False
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, x_train, y_train, x_test, y_test, checkpoint_path):
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", verbose=1, save_best_only=True, save_weights_only=False, save_freq='epoch')
    earlystop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
    history = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test), verbose=1, callbacks=[checkpoint, earlystop])
    return history

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    y_pred_prob = model.predict(x_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    conf_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Confusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(class_report)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat=cm, figsize=(8, 7), class_names=["Normal", "Cataract"], show_normed=True)
    plt.style.use("ggplot")
    plt.show()

if __name__ == "__main__":
    file_path = r'C:\Users\Naman\Desktop\6th sem moocs\dataset\full_df.csv'
    dataset_dir = r'C:\Users\Naman\Desktop\6th sem moocs\dataset\preprocessed_images'
    image_size = 224

    df = load_data(file_path)
    cataract, normal = preprocess_data(df)

    dataset_cataract = create_dataset(cataract, 1, dataset_dir, image_size)
    dataset_normal = create_dataset(normal, 0, dataset_dir, image_size)

    x = np.array([i[0] for i in dataset_cataract + dataset_normal]).reshape(-1, image_size, image_size, 3)
    y = np.array([i[1] for i in dataset_cataract + dataset_normal])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = build_model(input_shape=(image_size, image_size, 3))
    history = train_model(model, x_train, y_train, x_test, y_test, checkpoint_path="vgg19.keras")
    evaluate_model(model, x_test, y_test)
