import os

# ==========================================
# GPU / BACKEND CONFIGURATION
# ==========================================
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
# Using the full training dataset but limited to 300 images per class
DATASET_PATH = r"c:\Users\Aris\Downloads\DTL.youtube\asl_alphabet_train\asl_alphabet_train"
MODEL_SAVE_PATH = "keras_model.keras"
LABELS_SAVE_PATH = "labels.txt"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
SAMPLES_PER_CLASS = 300 # User requested limit

print(f"Keras Backend: {keras.backend.backend()}")

# ==========================================
# DATA LOADING (Limited to 300 per class)
# ==========================================
print(f"Loading dataset (max {SAMPLES_PER_CLASS} samples per class)...")

def load_limited_dataset(path, limit=300):
    image_paths = []
    labels = []
    
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    classes.sort()
    
    for c in classes:
        class_path = os.path.join(path, c)
        files = [f for f in os.listdir(class_path) if f.lower().endswith('.jpg')]
        # Take only the first N samples
        subset = files[:limit]
        for f in subset:
            image_paths.append(os.path.join(class_path, f))
            labels.append(c)
        print(f"Loaded {len(subset)} images for class: {c}")
            
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    indexed_labels = [label_to_index[l] for l in labels]
    
    with open(LABELS_SAVE_PATH, "w") as f:
        for label in unique_labels:
            f.write(f"{label}\n")
    
    return image_paths, indexed_labels, unique_labels

image_paths, labels_indexed, class_names = load_limited_dataset(DATASET_PATH, limit=SAMPLES_PER_CLASS)

# Split data (80% train, 20% validation)
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    image_paths, labels_indexed, test_size=0.2, random_state=42, stratify=labels_indexed
)

def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    label = tf.one_hot(label, len(class_names))
    return img, label

# Create tf.data.Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
train_ds = train_ds.shuffle(len(X_train_paths)).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val))
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
# MODEL (Professional Configuration)
# ==========================================
print("Building MobileNetV2 Model...")

base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Professional head with Dropout
inputs = keras.Input(shape=(224, 224, 3))
# Data Augmentation
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomTranslation(0.1, 0.1)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# TRAINING
# ==========================================
checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print(f"Starting training on {len(X_train_paths)} images...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ==========================================
# EVALUATION
# ==========================================
print("Evaluating model...")
def plot_history(h):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(h.history['accuracy'], label='train'); plt.plot(h.history['val_accuracy'], label='val'); plt.legend(); plt.title('Accuracy')
    plt.subplot(1, 2, 2); plt.plot(h.history['loss'], label='train'); plt.plot(h.history['val_loss'], label='val'); plt.legend(); plt.title('Loss')
    plt.show()

plot_history(history)
print(f"Done. Model saved to {MODEL_SAVE_PATH}")
