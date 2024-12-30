Title: E-Commerce Fashion Recommendation System (AI_570_Project)

The primary objective of our project is to develop a multi-modal recommendation system that utilizes deep learning techniques to enhance the accuracy and relevance of product recommendations in an e-commerce platform. By Improving the relevance of recommendations by leveraging both visual and textual product data—this system aims to create a comprehensive understanding of user preferences and behaviors.

-All Datasets and Models & Weights used in the code should be stored in the Model Folder
-You can always retrain the models with your own weights, if you would like to start from scratch to do your own optimization

Dataset Source:
We used the Fashion Product Images (Small) dataset from Kaggle.
Link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

The dataset consists of:

Images: Product images located in the images/ folder, identified by their IDs (e.g., images/42431.jpg).
Metadata: Textual information about products, including attributes such as:
productDisplayName
masterCategory
subCategory
baseColor
gender
season

Note: You will need to download the original dataset from link above and reorganize the downloaded dataset to match this folder and file structure:

Overall_Project_Folder/
    ├── main_script.py  (main Python script)
    │       
    ├── Datasets/			Main Original Dataset Data from Kaggle
    │       images/         (folder containing images)
    │       styles.csv      (metadata file)
    |
    |___ Input_Images/      (folder for user-uploaded images for similarity search)
    |
    |___ Model/     	    (Model and weights folder - weights for our final model CLIP are stored here)  
    |
    |
    |___ User_FormatedDatasets/	(If you want to save the datasets that get preprocessed for each model place them here the code already has this path established for such task)  
     