# Text Sentiment Analysis using Naive Bayes and Neural Network Classifier Models

## **I. Project Description**
This project aims to classify text data into positive and negative sentiments using two different machine learning models: Naive Bayes classifier and a Neural Network model. The project involves data preprocessing, model training, evaluation, and comparison of both models.

### **Project Goals**
- **Input**: Preprocessed text data.
- **Output**: Sentiment classification (positive or negative).

## **II. Introduction**
The objective is to build and evaluate two text classification models. The Naive Bayes classifier provides a probabilistic approach, while the Neural Network model uses embeddings for more nuanced text representations.

## **III. Dataset Information**
For this project, we are using the NTC-SCV dataset, which is available to access via this [GitHub](https://github.com/congnghia0609/ntc-scv) repository. The NTC-SCV dataset is a collection of Vietnamese text reviews used for sentiment analysis. It contains 50,000 samples, each labeled as either positive or negative. The dataset is specifically designed to help train and evaluate text classification models.

## **IV. Model Description**
### **1. Naive Bayes Classifier**
A probabilistic model based on Bayes' theorem, suitable for text classification tasks. It is simple, fast, and effective for high-dimensional datasets.

### **2. Neural Network Model**
A model consisting of an `EmbeddingBag` layer followed by a linear layer for classification. This model captures more complex patterns and relationships in the text data.

## **V. Model Implementation**

### **1. Naive Bayes Classifier**
- **Feature Extraction**: Using `CountVectorizer` to transform text data into numerical features.
- **Model Training**: Fitting a Naive Bayes classifier to the training data.
- **Evaluation**: Using metrics like accuracy, confusion matrix, and F1 score.

### **2. Neural Network Model**
- **Embedding Layer**: Converts words into vector representations.
- **Linear Layer**: Transforms the embeddings into class scores.
- **Training**: Using a training loop to optimize the model with Stochastic Gradient Descent (SGD).
- **Evaluation**: Similar to the Naive Bayes classifier, using accuracy, confusion matrix, and F1 score.

## **VI. Evaluation**
Both models are evaluated using accuracy, confusion matrix, and F1 score metrics. 
### 1. Confusion Matrix
Displays the number of correct and incorrect predictions, providing insight into the types of errors made by the model.

### 2. Classification Report
Includes precision, recall, and F1 score for each class, offering a detailed performance analysis.

### 3. F1 Score
The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

## **VII. Results**
* **Naive Bayes Classifier**: Achieved an accuracy of approximately 85%.
* **Neural Network Model**: Achieved an accuracy of 87.71%.

Despite the neural network model's slightly better performance, further improvements could be made by experimenting with more advanced models and hyperparameter tuning.

## **VIII. Usage**
Run the Jupyter notebook `sentiment_analysis.ipynb` to train and evaluate the models:
```bash
jupyter notebook sentiment_analysis.ipynb
```

### **Example Commands**
1. **Training the Naive Bayes Classifier**:
   ```python
   # Train the Naive Bayes Classifier
   nb_classifier.fit(X_train, y_train)
   ```

2. **Evaluating the Neural Network Model**:
   ```python
   # Evaluate the Neural Network Model
   eval_acc, eval_loss = evaluate(model, criterion, valid_dataloader)
   ```

## **IX. Conclusion**
The neural network model showed a marginally better performance compared to the Naive Bayes classifier. Future improvements could include experimenting with more advanced models, utilizing pre-trained embeddings, and performing hyperparameter tuning to enhance the model's accuracy further.

## **X. Dependencies**
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Sklearn
- Torch
- Torchvision

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## **XI. Contact**
For any questions or issues, please contact Uyen Nguyen via [nguyen_u1@denison.edu](mailto:nguyen_u1@denison.edu).

