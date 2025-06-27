# Amazon Review Sentiment Analysis using PySpark and TensorFlow

This project performs sentiment analysis on a large dataset of Amazon reviews. The primary goal is to build a model that can accurately classify the sentiment of a review as either positive or negative. The implementation leverages PySpark for efficient processing of large-scale data and TensorFlow/Keras for building a deep learning model.

## Project Overview

The project follows a standard machine learning workflow:

1.  **Data Loading and Preprocessing:** The Amazon Reviews dataset is loaded into a PySpark DataFrame. The text data (review title and text) is then combined and cleaned.
2.  **Feature Engineering:** A sophisticated feature engineering pipeline is constructed using PySpark's MLlib. This pipeline tokenizes the text, removes stop words, and generates both unigrams and bigrams. These n-grams are then converted into numerical features using `CountVectorizer` and weighted using `TF-IDF`.
3.  **Model Training:** A neural network is built using TensorFlow's Keras API. The model consists of dense layers with ReLU activation functions and a final sigmoid layer for binary classification.
4.  **Model Evaluation:** The trained model is evaluated on a test set. The performance is assessed using various metrics, including accuracy, precision, recall, and F1-score. Additionally, a confusion matrix and an ROC curve are generated to visualize the model's performance.

## Technologies and Libraries Used

* **PySpark:** For distributed data processing and building the feature engineering pipeline.
* **TensorFlow & Keras:** For creating and training the deep learning model.
* **Pandas:** For data manipulation.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization, including plotting the model's accuracy, loss, confusion matrix, and ROC curve.
* **Kaggle Hub:** For downloading the dataset.

## Dataset

The project uses the "Amazon Reviews" dataset, which is available on Kaggle. This dataset contains millions of reviews, each with a polarity (1 for negative, 2 for positive), a title, and the review text.

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rubricate12/amazonreview_withspark
    ```
2.  **Install the dependencies:**
    Make sure you have Python 3 and the following libraries installed:
    ```bash
    pip install pyspark tensorflow pandas numpy matplotlib seaborn kaggle
    ```
3.  **Download the dataset:**
    The notebook uses the `kagglehub` library to download the dataset. You will need to have your Kaggle API token configured.
4.  **Execute the Jupyter Notebook:**
    Run the `tubes-bigdat_final.ipynb` notebook to see the full process of data loading, preprocessing, model training, and evaluation.

## Results

The model achieves a high accuracy in classifying the sentiment of the Amazon reviews. The evaluation includes:
* **Accuracy and Loss Plots:** Visualizations of the training and validation accuracy and loss over epochs.
* **Classification Report:** A detailed report with precision, recall, and F1-score for both positive and negative classes.
* **Confusion Matrix:** A heatmap that visualizes the model's performance in terms of true positives, true negatives, false positives, and false negatives.
* **ROC Curve:** A plot of the true positive rate against the false positive rate, with the Area Under the Curve (AUC) calculated to show the model's ability to distinguish between classes.

This comprehensive evaluation demonstrates the effectiveness of the combined PySpark and TensorFlow approach for large-scale sentiment analysis.
