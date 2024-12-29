"""
Title: E-Commerce Fashion Recommendation System
Team 3: Rachel Brooks, Eric Brown, Claudio Mema
Course: AI 570 - Deep Learning (Fall 2024 Semester)

0.1.1 Problem Statement
The primary objective of our project is to develop a multi-modal recommendation system that utilizes deep learning techniques to enhance the accuracy and relevance of product recommendations in an e-commerce platform. By combining diverse data types—such as images, textual descriptions, and contextual information.

Keywords: E-Commerce Fashion Recommendation System

0.1.2 Data Collection
Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
Short Description: Each product image is identified by an ID like 42431. You will find a map to all the products in styles.csv. From here, you can fetch the image for this product from images/42431.jpg. To get started easily, we also have exposed some of the key product categories and it's display name in styles.csv.

Keywords: Fashion Product Images Dataset

0.1.3 Listed below are the required packages please check your local environment to see if you have them installed already
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pickle
import torch
from itertools import cycle
from PIL import Image
from collections import Counter
from transformers import CLIPProcessor, CLIPModel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

"""
0.1.4 Data Preprocessing
"""
#importing data
metadata = pd.read_csv('Datasets/styles.csv')
#### Data Collection - Exploratory Analysis ######
#Overall Data Snapshot
print("Data Snapshot:")
print(metadata.head())
print ("Feature Descriptions:")
print(metadata.info())

# Select the relevant fields for analysis
columns_of_interest = ['baseColour', 'masterCategory', 'subCategory', 'season', 'gender', 'articleType']
metadata2 = metadata[columns_of_interest]

#Relevant Columns Data Snapshot
print("Data Snapshot:")
print(metadata2.head())

#Missing Value Analysis
print("\nMissing Values:")
missing_values = metadata2.isnull().sum()
print(missing_values[missing_values > 0])

#Overall Data Structure
print(metadata.info())
print(metadata['masterCategory'].value_counts())
print(metadata['subCategory'].value_counts())

#Feature Distributions
# Limit the chart to the top 10 most frequent article types
top_n = 10
article_counts = metadata['articleType'].value_counts()
top_articles = article_counts[:top_n]
other_articles_count = article_counts[top_n:].sum()

# Create a new DataFrame for visualization
article_chart_data = pd.concat([top_articles, pd.Series({"Other": other_articles_count})])

# Plot the chart
plt.figure(figsize=(12, 8))
sns.barplot(x=article_chart_data.values, y=article_chart_data.index, palette="viridis")
plt.title(f"Top {top_n} Article Types (Others Grouped)")
plt.xlabel("Count")
plt.ylabel("Article Type")
plt.show()

#Cross-tabulation for Relationships
plt.figure(figsize=(12, 8))
gender_article_ct = pd.crosstab(metadata['gender'], metadata['articleType'])
gender_article_ct.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title("Gender vs Article Type")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Article Type")
plt.show()

#Season Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='season', hue='gender', data=metadata)
plt.title("Season-wise Distribution by Gender")
plt.xlabel("Season")
plt.ylabel("Count")
plt.show()

# Descriptive statistics for continuous features
print("Descriptive Statistics for Continuous Features:")
print(metadata.describe())

# Skewness of continuous features
continuous_features = metadata.select_dtypes(include=['float64', 'int64']).columns
print("\nSkewness of Continuous Features:")
for col in continuous_features: #only year variable is continuous
    print(f"{col}: {metadata[col].skew()}")

# Range calculation
ranges = metadata[continuous_features].max() - metadata[continuous_features].min()
print("\nRange of Continuous Features:")
print(ranges)

# Convert categorical columns to object type to ensure compatibility
categorical_columns = ['baseColour', 'masterCategory', 'subCategory', 'season', 'gender', 'articleType']
metadata[categorical_columns] = metadata[categorical_columns].astype("object")

# Fill missing values with "Unknown" for categorical variables
metadata[categorical_columns] = metadata[categorical_columns].fillna("Unknown")

# Fill missing values in 'season' with its mode
season_mode = metadata['season'].mode()[0]  # Compute the mode
metadata['season'] = metadata['season'].fillna(season_mode)  # Use fillna without inplace

# Confirm the operation
print(metadata.isnull().sum())  # Verify no missing values remain

#remove remaining missing values and columns
# Remove rows with missing productDisplayName
metadata = metadata[metadata['productDisplayName'].notnull()]

# Fill missing values in usage with "Unknown"
metadata.loc[:, 'usage'] = metadata['usage'].fillna('Unknown')

# Drop the single record with a missing year
metadata = metadata[metadata['year'].notnull()]

# Drop redundant columns
metadata = metadata.drop(columns=['Unnamed: 10', 'Unnamed: 11'])


# Grouping rare subcategories
subCategory_counts = metadata['subCategory'].value_counts()
rare_threshold = 19  # Adjust threshold
metadata['subCategory'] = metadata['subCategory'].apply(
    lambda x: x if subCategory_counts[x] > rare_threshold else 'Other'
)
print(metadata['subCategory'].value_counts())

# Filter out rows where 'masterCategory' is 'Home'
metadata = metadata[metadata['masterCategory'] != 'Home']

# Confirm the category has been removed
print(metadata['masterCategory'].value_counts())

# Detect inconsistent values in categorical columns
for col in categorical_columns:
    print(f"\nUnique values in {col}: {metadata[col].unique()}")

###############################################################################
#show five images from dataset
for x in range(5):
    sample = metadata.iloc[x]
    img = Image.open(f"Datasets/images/{sample['id']}.jpg")
    plt.imshow(img)
    plt.title(sample['productDisplayName'])
    plt.show()


# Image augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,  # Randomly rotate images by ±20 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally
    height_shift_range=0.1,  # Randomly shift images vertically
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flips
    fill_mode='nearest'  # Fill empty pixels after transformations
)


# Function to preprocess and augment an image
def preprocess_and_augment_image(image):
    # Load and preprocess image
    img = load_img(image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    # Augment image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for augmentation
    augmented_iter = datagen.flow(img_array, batch_size=1)  # Apply augmentation
    augmented_image = next(augmented_iter)[0]  # Get the augmented image
    return augmented_image


# Check if file paths exist before processing and augmenting
def safe_preprocess(row):
    if os.path.exists(row['image_path']):
        return preprocess_and_augment_image(row['image_path'])
    else:
        print(f"File not found: {row['image_path']}")
        return None


# Preprocess and augment all images in the dataset
metadata.loc[:, 'image_path'] = metadata['id'].apply(
    lambda x: os.path.normpath(os.path.join(image_dir, f"{x}.jpg"))
)

# Verify file paths and preprocess/augment images
metadata.loc[:, 'image_data'] = metadata.apply(safe_preprocess, axis=1)

# Drop rows where images were missing
metadata = metadata[metadata['image_data'].notnull()].reset_index(drop=True)
print(f"Remaining records after dropping missing images: {metadata.shape[0]}")

# Convert to NumPy array for model training
image_data = np.array(metadata['image_data'].tolist())
print(f"Shape of preprocessed and augmented image data: {image_data.shape}")

#save image data for later use
with open('User_FormatedDatasets/image_data.pkl','wb') as file:
    pickle.dump(image_data,file)
#save dataset for future use
metadata.to_pickle("User_FormatedDatasets/metadata.pkl")
# Preprocessing is over now, model building begins
#################################################################################
"""
0.1.5 Methodology
Model 1:
The ResNet Multi-Task Model was selected because it efficiently processes visual data and performs multi-label classification. Its architecture is well-suited for capturing detailed features in fashion images, making it ideal for e-commerce applications

Base Architecture: ResNet (Residual Network) for image feature extraction.
Output Structure: Multi-task setup with six independent output layers to predict different attributes such as articleType, baseColor, gender, masterCategory, season, and subCategory.


Model 2:
The CLIP (Contrastive Language-Image Pretraining) model was chosen for its ability to process both textual and visual data simultaneously. It was designed for embedding-based similarity search, making it highly flexible for recommendations beyond strict classification.

Base Architecture: CLIP’s vision and text encoders were used to generate embeddings for images and product descriptions.
Embedding Combination: A weighted average of image and text embeddings was computed to create a unified representation for each product.
"""
#read in data
metadata = pd.read_pickle("User_FormatedDatasets/metadata.pkl")
with open('User_FormatedDatasets/image_data.pkl','rb') as file:
    image_data = pickle.load(file)

# Define an input layer
input_layer = Input(shape=(224, 224, 3))  # Assuming image data has this shape

#set the seed for reproducibility
seed = 38
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Encode labels
def encode_labels(column):
    return to_categorical(LabelEncoder().fit_transform(metadata[column]))

master_labels = encode_labels('masterCategory')
sub_labels = encode_labels('subCategory')
articleType_labels = encode_labels('articleType')
season_labels = encode_labels('season')
baseColor_labels = encode_labels('baseColour')
gender_labels = encode_labels('gender')

#training and test split
X_train, X_val, y_train_master, y_val_master, y_train_sub, y_val_sub, y_train_articleType, y_val_articleType, y_train_season, y_val_season, y_train_baseColor, y_val_baseColor, y_train_gender, y_val_gender = train_test_split(
    image_data,
    master_labels,
    sub_labels,
    articleType_labels,
    season_labels,
    baseColor_labels,
    gender_labels,
    test_size=0.3,
    random_state=seed
)


# Unpack labels for masterCategory
y_train_master_single = np.argmax(y_train_master, axis=1)  # Convert one-hot encoded labels to single-label format
y_val_master_single = np.argmax(y_val_master, axis=1)

# Calculate class distributions for training and validation sets
unique_train, counts_train = np.unique(y_train_master_single, return_counts=True)
unique_val, counts_val = np.unique(y_val_master_single, return_counts=True)

# Plot class distribution
plt.bar(unique_train, counts_train, alpha=0.7, label='Training Set')
plt.bar(unique_val, counts_val, alpha=0.7, label='Validation Set', color='orange')

plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training and Validation Sets (masterCategory)')
plt.legend()
plt.show()

# Unpack labels for subCategory
y_train_sub_single = np.argmax(y_train_sub, axis=1)  # Convert one-hot encoded labels to single-label format
y_val_sub_single = np.argmax(y_val_sub, axis=1)

# Calculate class distributions for training and validation sets
unique_train, counts_train = np.unique(y_train_sub_single, return_counts=True)
unique_val, counts_val = np.unique(y_val_sub_single, return_counts=True)

# Plot class distribution
plt.bar(unique_train, counts_train, alpha=0.7, label='Training Set')
plt.bar(unique_val, counts_val, alpha=0.7, label='Validation Set', color='orange')

plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training and Validation Sets (subCategory)')
plt.legend()
plt.show()

# Unpack labels for articleType
y_train_art_single = np.argmax(y_train_articleType, axis=1)  # Convert one-hot encoded labels to single-label format
y_val_art_single = np.argmax(y_val_articleType, axis=1)

# Calculate class distributions for training and validation sets
unique_train, counts_train = np.unique(y_train_art_single, return_counts=True)
unique_val, counts_val = np.unique(y_val_art_single, return_counts=True)

# Plot class distribution
plt.bar(unique_train, counts_train, alpha=0.7, label='Training Set')
plt.bar(unique_val, counts_val, alpha=0.7, label='Validation Set', color='orange')

plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training and Validation Sets (articleType)')
plt.legend()
plt.show()

# Unpack labels for season
y_train_season_single = np.argmax(y_train_season, axis=1)  # Convert one-hot encoded labels to single-label format
y_val_season_single = np.argmax(y_val_season, axis=1)

# Calculate class distributions for training and validation sets
unique_train, counts_train = np.unique(y_train_season_single, return_counts=True)
unique_val, counts_val = np.unique(y_val_season_single, return_counts=True)

# Plot class distribution
plt.bar(unique_train, counts_train, alpha=0.7, label='Training Set')
plt.bar(unique_val, counts_val, alpha=0.7, label='Validation Set', color='orange')

plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training and Validation Sets (season)')
plt.legend()
plt.show()

# Unpack labels for gender
y_train_gender_single = np.argmax(y_train_gender, axis=1)  # Convert one-hot encoded labels to single-label format
y_val_gender_single = np.argmax(y_val_gender, axis=1)

# Calculate class distributions for training and validation sets
unique_train, counts_train = np.unique(y_train_gender_single, return_counts=True)
unique_val, counts_val = np.unique(y_val_gender_single, return_counts=True)

# Plot class distribution
plt.bar(unique_train, counts_train, alpha=0.7, label='Training Set')
plt.bar(unique_val, counts_val, alpha=0.7, label='Validation Set', color='orange')

plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training and Validation Sets (Gender)')
plt.legend()
plt.show()


###### use weight adj in model discoved from view of data from training and validation dataset split ########
"""
0.1.6 Model Fitting and Validation - Model 1
"""
# Feature extraction with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer, pooling='avg')

# Allow all layers to be trainable
for layer in base_model.layers:
    layer.trainable = True

# Shared layers
x = Dense(512, activation='relu')(base_model.output)
x = Dropout(0.4)(x)

# Task-specific layers
master_output = Dense(master_labels.shape[1], activation='softmax', name='master_output')(x)
sub_output = Dense(sub_labels.shape[1], activation='softmax', name='sub_output')(x)
articleType_output = Dense(articleType_labels.shape[1], activation='softmax', name='articleType_output')(x)
season_output = Dense(season_labels.shape[1], activation='softmax', name='season_output')(x)
baseColor_output = Dense(baseColor_labels.shape[1], activation='softmax', name='baseColor_output')(x)
gender_output = Dense(gender_labels.shape[1], activation='softmax', name='gender_output')(x)

# Model definition
multi_task_model = Model(
    inputs=input_layer,
    outputs=[master_output, sub_output, articleType_output, season_output, baseColor_output, gender_output]
)

# Compile the model
multi_task_model.compile(
    optimizer=SGD(learning_rate=1e-4, momentum=0.9, nesterov=True),
    loss={
        'master_output': 'categorical_crossentropy',
        'sub_output': 'categorical_crossentropy',
        'articleType_output': 'categorical_crossentropy',
        'season_output': 'categorical_crossentropy',
        'baseColor_output': 'categorical_crossentropy',
        'gender_output': 'categorical_crossentropy'
    },
    loss_weights={
        'master_output': 1.0,  # Balanced
        'sub_output': 1.5,  # Slightly higher for imbalanced subCategory
        'articleType_output': 2.0,  # Higher for imbalanced articleType
        'season_output': 1.0,  # Balanced
        'baseColor_output': 1.0,  # Balanced
        'gender_output': 1.0  # Balanced
    },
    metrics={
        'master_output': 'accuracy',
        'sub_output': 'accuracy',
        'articleType_output': 'accuracy',
        'season_output': 'accuracy',
        'baseColor_output': 'accuracy',
        'gender_output': 'accuracy'
    }
)

# Prepare training and validation labels
train_labels = {
    'master_output': y_train_master,  # Original labels for masterCategory
    'sub_output': y_train_sub,  # Original labels for subCategory
    'articleType_output': y_train_articleType,  # Original labels for articleType
    'season_output': y_train_season,  # Original labels for season
    'baseColor_output': y_train_baseColor,  # Original labels for baseColor
    'gender_output': y_train_gender  # Original labels for gender
}

val_labels = {
    'master_output': y_val_master,
    'sub_output': y_val_sub,
    'articleType_output': y_val_articleType,
    'season_output': y_val_season,
    'baseColor_output': y_val_baseColor,
    'gender_output': y_val_gender
}

####### Model Training #######
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = multi_task_model.fit(
    X_train,
    train_labels,
    validation_data=(X_val, val_labels),
    epochs=2,
    batch_size=16,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

"""
0.1.7 Model Evaluation - Model 1
"""
def plot_training_history(history):
    # List all the metrics
    metrics = ['loss', 'master_output_accuracy', 'sub_output_accuracy',
               'articleType_output_accuracy', 'season_output_accuracy',
               'baseColor_output_accuracy', 'gender_output_accuracy']

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric in history.history:
            axes[i].plot(history.history[metric], label='Train')
            axes[i].plot(history.history[f'val_{metric}'], label='Validation')
            axes[i].set_title(f'Model {metric.capitalize()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
        else:
            axes[i].set_title(f'{metric.capitalize()} (Metric Missing)')
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# Call the function with your training history
plot_training_history(history)

#steps to get top five similar items in dataset compared with own input photo
# Define an embedding model to extract features
embedding_layer = multi_task_model.get_layer('dense')  # Shared Dense layer
embedding_model = Model(inputs=multi_task_model.input, outputs=embedding_layer.output)


##### Now view predictions of model in conjunction with user input photo to get similarity score between both results #####
# Generate embeddings for the dataset
dataset_features = embedding_model.predict(X_train, batch_size=32)

# Normalize dataset features
dataset_features = normalize(dataset_features)

# Function to preprocess a single input image
def preprocess_input_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to match model input size
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply ResNet50 preprocessing
    return np.expand_dims(img_array, axis=0)

# Load and preprocess the input image
input_image_path = 'Input_Images/0452.jpg'   #you can use your own photos just make sure you save it in the input images folder in this directory structure
#   - Overall Project folder
#   - code file
#   - input folder inside of overall project folder for input images
input_image = preprocess_input_image(input_image_path)

# Generate embedding for the input image
input_features = embedding_model.predict(input_image)

# Normalize the input features
input_features = normalize(input_features)

# Compute cosine similarity between input image and dataset features
similarities = cosine_similarity(input_features, dataset_features).flatten()

# Retrieve indices of the top-5 similar items
top_n_indices = similarities.argsort()[-5:][::-1]  # Sort and get top-5 indices in descending order

# Display the top-5 similar items
for idx in top_n_indices:
    print(f"Similarity Score: {similarities[idx]:.4f}")
    product_name = metadata.iloc[idx]['productDisplayName']  # Replace with the appropriate column
    image_path = metadata.iloc[idx]['image_path']  # Replace with the appropriate column
    print(f"Recommended Item: {product_name}")
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(product_name)
    plt.axis('off')
    plt.show()

# Plot distribution of similarity scores
plt.hist(similarities, bins=50, alpha=0.7)
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.title("Distribution of Cosine Similarity Scores")
plt.show()

######### Debugging Code  Below for understanding model problems ###########
# Predictions
predictions = multi_task_model.predict(X_val, verbose=1)

# Extract true and predicted labels for articleType
# Assuming label_encoder was used during preprocessing
label_encoder = LabelEncoder()
label_encoder.fit(metadata['articleType'])

# Decode numerical labels back to string labels
articleType_true_labels = label_encoder.inverse_transform(np.argmax(y_val_articleType, axis=1))
articleType_pred = np.argmax(predictions[2], axis=1)
articleType_pred_labels = label_encoder.inverse_transform(articleType_pred)


print("Classification Report for ArticleType:")
print(classification_report(articleType_true_labels, articleType_pred_labels))


# Extract true and predicted labels for baseColor
# Assuming label_encoder was used during preprocessing
label_encoder = LabelEncoder()
label_encoder.fit(metadata['baseColour'])

# Decode numerical labels back to string labels
baseColor_true_labels = label_encoder.inverse_transform(np.argmax(y_val_baseColor, axis=1))
baseColor_pred = np.argmax(predictions[4], axis=1)
baseColor_pred_labels = label_encoder.inverse_transform(baseColor_pred)

print("Classification Report for BaseColour:")
print(classification_report(baseColor_true_labels, baseColor_pred_labels))

embedding_model = Model(inputs=multi_task_model.input, outputs=multi_task_model.get_layer('dense').output)
embeddings = embedding_model.predict(X_val)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plot clusters with labels
plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=articleType_true, palette='viridis')
plt.title('PCA Visualization of ArticleType Clusters')
plt.show()
######### End of Debugging ##############################

"""
0.1.6 Model Fitting and Validation - Model 2
"""
# Load the CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# Load metadata
metadata = pd.read_pickle("User_FormatedDatasets/metadata.pkl")

# Set batch size
BATCH_SIZE = 64
seed = 38
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Image augmentation setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to preprocess and augment a single image for CLIP
def preprocess_single_image_for_clip(image_path):
    # Load image
    img = load_img(image_path, target_size=(224, 224))  # Resize to CLIP's required size
    img_array = img_to_array(img)

    # Normalize to [0, 1]
    img_array = img_array / 255.0

    # Augment image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for augmentation
    augmented_iter = datagen.flow(img_array, batch_size=1)  # Apply augmentation
    augmented_image = next(augmented_iter)[0]  # Get the augmented image

    # Convert to PIL-compatible format (values in [0, 255])
    return Image.fromarray((augmented_image * 255).astype(np.uint8))


# Function to process images and text in batches
def generate_embeddings_in_batches(metadata, clip_processor, batch_size):
    num_batches = len(metadata) // batch_size + int(len(metadata) % batch_size > 0)
    image_embeddings_list = []
    text_embeddings_list = []

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(metadata))

            batch_metadata = metadata.iloc[start_idx:end_idx]

            # Process batch images
            batch_images = [
                preprocess_single_image_for_clip(row['image_path']) for _, row in batch_metadata.iterrows()
            ]

            # Convert images and texts into CLIP inputs
            batch_texts = batch_metadata['productDisplayName'].tolist()
            batch_inputs = clip_processor(
                text=batch_texts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Generate embeddings
            batch_image_embeddings = clip_model.get_image_features(pixel_values=batch_inputs["pixel_values"])
            batch_text_embeddings = clip_model.get_text_features(input_ids=batch_inputs["input_ids"],
                                                                 attention_mask=batch_inputs["attention_mask"])

            # Append embeddings
            image_embeddings_list.append(batch_image_embeddings)
            text_embeddings_list.append(batch_text_embeddings)

    # Concatenate all batches
    image_embeddings = torch.cat(image_embeddings_list, dim=0)
    text_embeddings = torch.cat(text_embeddings_list, dim=0)

    return image_embeddings, text_embeddings

# Generate embeddings in batches
image_embeddings, text_embeddings = generate_embeddings_in_batches(metadata, clip_processor, batch_size=BATCH_SIZE)

# Combine embeddings with weights
text_weight = 0.65
image_weight = 0.35
combined_embeddings = (text_weight * text_embeddings + image_weight * image_embeddings)

# Save the embeddings
torch.save(image_embeddings, "Model/image_embeddings.pt")
torch.save(text_embeddings, "Model/text_embeddings.pt")
torch.save(combined_embeddings, "Model/combined_embeddings.pt")

# Encode labels
def encode_labels(column):
    le = LabelEncoder()
    encoded = le.fit_transform(metadata[column])
    return to_categorical(encoded), le

master_labels, master_encoder = encode_labels('masterCategory')
sub_labels, sub_encoder = encode_labels('subCategory')
articleType_labels, articleType_encoder = encode_labels('articleType')
season_labels, season_encoder = encode_labels('season')
baseColor_labels, baseColor_encoder = encode_labels('baseColour')
gender_labels, gender_encoder = encode_labels('gender')

# Save all encoders in a single file
encoders = {
    "articleType": articleType_encoder,
    "baseColor": baseColor_encoder,
    "gender": gender_encoder,
    "master": master_encoder,
    "season": season_encoder,
    "sub": sub_encoder
}
# Save encoders for future use
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Train-test split
(X_train, X_test, y_train_master, y_test_master, y_train_sub, y_test_sub,
 y_train_articleType, y_test_articleType, y_train_season, y_test_season,
 y_train_baseColor, y_test_baseColor, y_train_gender, y_test_gender
 ) = train_test_split(
    combined_embeddings.numpy(), master_labels, sub_labels, articleType_labels,
    season_labels, baseColor_labels, gender_labels, test_size=0.3, random_state=seed
)

# Build multi-task model
input_layer = Input(shape=(512,))
x = Dense(512, activation='relu')(input_layer)
x = Dropout(0.4)(x)

master_output = Dense(master_labels.shape[1], activation='softmax', name='master_output')(x)
sub_output = Dense(sub_labels.shape[1], activation='softmax', name='sub_output')(x)
articleType_output = Dense(articleType_labels.shape[1], activation='softmax', name='articleType_output')(x)
season_output = Dense(season_labels.shape[1], activation='softmax', name='season_output')(x)
baseColor_output = Dense(baseColor_labels.shape[1], activation='softmax', name='baseColor_output')(x)
gender_output = Dense(gender_labels.shape[1], activation='softmax', name='gender_output')(x)

multi_task_model = Model(inputs=input_layer, outputs=[
    master_output, sub_output, articleType_output, season_output, baseColor_output, gender_output
])

multi_task_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={
        'master_output': 'categorical_crossentropy',
        'sub_output': 'categorical_crossentropy',
        'articleType_output': 'categorical_crossentropy',
        'season_output': 'categorical_crossentropy',
        'baseColor_output': 'categorical_crossentropy',
        'gender_output': 'categorical_crossentropy'
    },
    loss_weights={
        'master_output': 1.0,
        'sub_output': 1.5,
        'articleType_output': 4.5,
        'season_output': 1.5,
        'baseColor_output': 4.5,
        'gender_output': 1.0
    },
    metrics={
        'master_output': 'accuracy',
        'sub_output': 'accuracy',
        'articleType_output': 'accuracy',
        'season_output': 'accuracy',
        'baseColor_output': 'accuracy',
        'gender_output': 'accuracy'
    }
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = multi_task_model.fit(
    X_train,
    [y_train_master, y_train_sub, y_train_articleType, y_train_season, y_train_baseColor, y_train_gender],
    validation_split=0.2,
    epochs=25,
    batch_size=64,
    callbacks=[reduce_lr, early_stopping]
)

#Save the model
multi_task_model.save('Model/multi_task_model.h5')
multi_task_model.save('Model/multi_task_model.keras')
multi_task_model.save_weights('Model/multi_task_model_weights.weights.h5')

# Define a color cycle for better distinction between plots
colors = cycle(plt.cm.tab20.colors)


"""
0.1.7 Model Evaluation - Model 2
"""
# Plot accuracy for each output
plt.figure(figsize=(12, 8))
for key in history.history.keys():
    if 'accuracy' in key:  # Filter accuracy keys
        plt.plot(history.history[key], label=key, color=next(colors))
plt.title('Model 2 Accuracy by Output', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Reset the color cycle for loss plots
colors = cycle(plt.cm.tab20.colors)

# Plot loss for each output
plt.figure(figsize=(12, 8))
for key in history.history.keys():
    if 'loss' in key:  # Filter loss keys
        plt.plot(history.history[key], label=key, color=next(colors))
plt.title('Model 2 Loss by Output', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# Cluster the combined embeddings
n_clusters = len(metadata['articleType'].unique())  # Number of clusters = unique article types
kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
metadata['cluster'] = kmeans.fit_predict(combined_embeddings.detach().numpy())

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(combined_embeddings.detach().numpy())

plt.figure(figsize=(10, 8))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=metadata['cluster'], palette='viridis')
plt.title('PCA Visualization of Clusters (CLIP)')
plt.show()

# Load encoders from the saved file
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Extract individual encoders
articleType_encoder = encoders["articleType"]
baseColor_encoder = encoders["baseColor"]
gender_encoder = encoders["gender"]
master_encoder = encoders["master"]
season_encoder = encoders["season"]
sub_encoder = encoders["sub"]

# Weights for combining embeddings
text_weight = 0.5
image_weight = 0.5

# Predict for single input
input_image_path = metadata.iloc[0]['image_path']
input_text = metadata.iloc[0]['productDisplayName']

# Preprocess a single image and text input for CLIP
def process_single_image_and_text(image_path, text, clip_processor):
    # Preprocess the image
    def preprocess_single_image_for_clip(image_path):
        # Load image
        img = load_img(image_path, target_size=(224, 224))  # Resize to CLIP's required size
        img_array = img_to_array(img)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Convert to PIL-compatible format (values in [0, 255])
        return Image.fromarray((img_array * 255).astype(np.uint8))

    # Preprocess the image
    processed_image = preprocess_single_image_for_clip(image_path)

    # Process image and text inputs using the CLIP processor
    image_inputs = clip_processor(images=processed_image, return_tensors="pt", padding=True, truncation=True)
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    return image_inputs, text_inputs

# Process the single image and text
image_inputs, text_inputs = process_single_image_and_text(input_image_path, input_text, clip_processor)

# Get embeddings from CLIP model
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])
    text_features = clip_model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])

# Combine embeddings with weights
combined_features = (text_weight * text_features + image_weight * image_features).cpu().numpy().reshape(1, -1)

# Predict using the multi-task model
predictions = multi_task_model.predict(combined_features)

# Safe decoding function
def safe_decode(predicted_index, encoder):
    try:
        if predicted_index < len(encoder.classes_):  # Check for valid index
            return encoder.inverse_transform([predicted_index])[0]
        else:
            return "Unknown"
    except ValueError:
        return "Unknown"

# Decode predictions for all outputs
decoded_articleType = safe_decode(np.argmax(predictions[0]), articleType_encoder)
decoded_baseColor = safe_decode(np.argmax(predictions[1]), baseColor_encoder)
decoded_gender = safe_decode(np.argmax(predictions[2]), gender_encoder)
decoded_master = safe_decode(np.argmax(predictions[3]), master_encoder)
decoded_season = safe_decode(np.argmax(predictions[4]), season_encoder)
decoded_sub = safe_decode(np.argmax(predictions[5]), sub_encoder)

# Print results
print(f"Predicted Article Type: {decoded_articleType}")
print(f"Predicted Base Color: {decoded_baseColor}")
print(f"Predicted Gender: {decoded_gender}")
print(f"Predicted Master Category: {decoded_master}")
print(f"Predicted Season: {decoded_season}")
print(f"Predicted Sub Category: {decoded_sub}")

##### Debugging Issue with classifiers decoded maybe incorrectly
# Function to visualize embeddings
def visualize_embeddings(embeddings, labels, method='pca', title='Embeddings Visualization', num_classes=10):
    """
    Visualize embeddings using PCA or t-SNE.

    Parameters:
        embeddings (numpy array): Embeddings to visualize.
        labels (array-like): Corresponding labels for coloring the embeddings.
        method (str): Visualization method - 'pca' or 'tsne'.
        title (str): Title for the plot.
        num_classes (int): Number of unique classes to display (for large datasets).
    """
    # Reduce dimensionality to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Get unique labels for plotting
    unique_labels = np.unique(labels)
    if len(unique_labels) > num_classes:
        unique_labels = unique_labels[:num_classes]

    plt.figure(figsize=(12, 8))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=label, alpha=0.6)

    plt.title(title, fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.5)
    plt.show()


# Convert embeddings to NumPy arrays
image_embeddings_np = image_embeddings.cpu().numpy()
text_embeddings_np = text_embeddings.cpu().numpy()
combined_embeddings_np = combined_embeddings.cpu().numpy()

# Use articleType labels for coloring (replace with the labels of your choice)
article_type_labels = metadata['articleType'].astype('category').cat.codes.values

# Visualize image embeddings
visualize_embeddings(
    embeddings=image_embeddings_np,
    labels=article_type_labels,
    method='pca',
    title='Image Embeddings (PCA)'
)

# Visualize text embeddings
visualize_embeddings(
    embeddings=text_embeddings_np,
    labels=article_type_labels,
    method='tsne',
    title='Text Embeddings (t-SNE)'
)

# Visualize combined embeddings
visualize_embeddings(
    embeddings=combined_embeddings_np,
    labels=article_type_labels,
    method='pca',
    title='Combined Embeddings (PCA)'
)


# Reduce combined embeddings to 2 dimensions using PCA
pca = PCA(n_components=2)
reduced_combined_embeddings = pca.fit_transform(combined_embeddings)

# Convert embeddings and metadata into a DataFrame for Plotly
def create_embedding_dataframe(embeddings, metadata, labels_column):
    df = pd.DataFrame(embeddings, columns=["Component 1", "Component 2"])
    df[labels_column] = metadata[labels_column].values
    return df

# Create DataFrame for reduced combined embeddings
combined_df = create_embedding_dataframe(reduced_combined_embeddings, metadata, "articleType")

fig_combined = px.scatter(
    combined_df,
    x="Component 1",
    y="Component 2",
    color="articleType",  # Change to your desired category like 'gender', 'masterCategory'
    title="Interactive Visualization of Combined Embeddings (PCA)",
    hover_data=["articleType"]  # Add more metadata columns here for interactivity
)

fig_combined.update_traces(marker=dict(size=5, opacity=0.7))
fig_combined.show()
################################################################################################################################################
# now using clip model's predictions in conjunction user model input image and text description to see how good model similarity predictions are
#################################################################################################################################################
# Predict for single input; again you can use your own photos just keep them in the folder as described earlier
input_image_path = 'Input_Images/0452.jpg'
#text decription of the photo which you decide to use
input_text = 'mens long sleeve shirt'

# Process the single image and text
image_inputs, text_inputs = process_single_image_and_text(input_image_path, input_text, clip_processor)

# Get embeddings from CLIP model
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])
    text_features = clip_model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])

# Combine embeddings with weights
combined_features = (text_weight * text_features + image_weight * image_features).cpu().numpy().reshape(1, -1)

# Compute cosine similarity between the input and all combined embeddings
similarities = cosine_similarity(combined_features, combined_embeddings)[0]

# Get the top 5 most similar items
top_indices = np.argsort(similarities)[::-1][:5]
top_similar_items = metadata.iloc[top_indices]

# Display the results
print("Top 5 Similar Items:")
img = Image.open(input_image_path)
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')
plt.show()

for idx, row in top_similar_items.iterrows():
    print(f"Item {idx + 1}:")
    print(f"  Product Name: {row['productDisplayName']}")
    print(f"  Article Type: {row['articleType']}")
    print(f"  Similarity Score: {similarities[idx]:.4f}")

    # Display the image with similarity score
    img = preprocess_single_image_for_clip(row['image_path'])
    plt.figure(figsize=(8, 8))  # Adjust figure size if needed
    plt.imshow(img)
    plt.title(f"Item {idx + 1}: {row['productDisplayName']}\n"
              f"Article Type: {row['articleType']}\n"
              f"Similarity Score: {similarities[idx]:.4f}")
    plt.axis('off')
    plt.show()
##########################################################################################################
# another similarity prediction analysis with another user input photo/image
##########################################################################################################
input_image_path = 'Input_Images/0452_1.jpg'
input_text = 'mens long sleeve shirt'

# Process the single image and text
image_inputs, text_inputs = process_single_image_and_text(input_image_path, input_text, clip_processor)

# Get embeddings from CLIP model
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])
    text_features = clip_model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])

# Combine embeddings with weights
combined_features = (text_weight * text_features + image_weight * image_features).cpu().numpy().reshape(1, -1)

# Compute cosine similarity between the input and all combined embeddings
similarities = cosine_similarity(combined_features, combined_embeddings)[0]
# Get the top 5 most similar items
top_indices = np.argsort(similarities)[::-1][:5]
top_similar_items = metadata.iloc[top_indices]

# Display the results
print("Top 5 Similar Items:")
img = Image.open(input_image_path)
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')
plt.show()

for idx, row in top_similar_items.iterrows():
    print(f"Item {idx + 1}:")
    print(f"  Product Name: {row['productDisplayName']}")
    print(f"  Article Type: {row['articleType']}")
    print(f"  Similarity Score: {similarities[idx]:.4f}")

    # Display the image with similarity score
    img = preprocess_single_image_for_clip(row['image_path'])
    plt.figure(figsize=(8, 8))  # Adjust figure size if needed
    plt.imshow(img)
    plt.title(f"Item {idx + 1}: {row['productDisplayName']}\n"
              f"Article Type: {row['articleType']}\n"
              f"Similarity Score: {similarities[idx]:.4f}")
    plt.axis('off')
    plt.show()
###########################################################################################
# another similarity prediction analysis with a photo from original image dataset
###########################################################################################
input_image_path = metadata.iloc[0]['image_path']
input_text = metadata.iloc[0]['productDisplayName']


# Process the single image and text
image_inputs, text_inputs = process_single_image_and_text(input_image_path, input_text, clip_processor)

# Get embeddings from CLIP model
with torch.no_grad():
    image_features = clip_model.get_image_features(pixel_values=image_inputs["pixel_values"])
    text_features = clip_model.get_text_features(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])

# Combine embeddings with weights
combined_features = (text_weight * text_features + image_weight * image_features).cpu().numpy().reshape(1, -1)

# Compute cosine similarity between the input and all combined embeddings
similarities = cosine_similarity(combined_features, combined_embeddings)[0]
# Get the top 5 most similar items
top_indices = np.argsort(similarities)[::-1][:5]
top_similar_items = metadata.iloc[top_indices]
# Display the results
print("Top 5 Similar Items:")
img = Image.open(input_image_path)
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')
plt.show()

for idx, row in top_similar_items.iterrows():
    print(f"Item {idx + 1}:")
    print(f"  Product Name: {row['productDisplayName']}")
    print(f"  Article Type: {row['articleType']}")
    print(f"  Similarity Score: {similarities[idx]:.4f}")

    # Display the image with similarity score
    img = preprocess_single_image_for_clip(row['image_path'])
    plt.figure(figsize=(8, 8))  # Adjust figure size if needed
    plt.imshow(img)
    plt.title(f"Item {idx + 1}: {row['productDisplayName']}\n"
              f"Article Type: {row['articleType']}\n"
              f"Similarity Score: {similarities[idx]:.4f}")
    plt.axis('off')
    plt.show()

"""
0.1.8 Issues/Improvements:

Feature-Specific Optimization: Fine-tuning the model to handle specific outputs like "season" more effectively, potentially by adding domain-specific embeddings or auxiliary features, could improve its accuracy.
Transfer Learning: Incorporating additional pretrained models specifically designed for the fashion domain might boost the model's understanding of subtle patterns and features unique to this industry.
"""