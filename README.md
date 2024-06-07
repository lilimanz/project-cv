![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project in Image Processing and Computer Vision

## Introduction
The field of computer vision is revolutionizing the way machines interact with the visual world. Through this project, students will engage with fundamental techniques in image classification, applying machine learning models to real-world datasets. This will provide hands-on experience in both developing and deploying classifiers that can discern between different categories of objects in images.

## Project Overview

This project is divided into several phases, each designed to deepen your understanding and skills in computer vision:

1. **Dataset Selection**: Choose one of the provided datasets for image classification. Each dataset has unique characteristics and challenges:
   - [Recycling](https://drive.google.com/file/d/1WhDq3xo2T-a8BAbx0ByoF8K1zvrHE5f2/view?usp=sharing): The dataset consists of images for machine learning classification, divided into two categories: recyclable and household waste. It aims to facilitate the development of models that can automate the sorting of these waste types, enhancing recycling processes.
   - [Bone Fractures](https://drive.google.com/file/d/1WeuxOenviI1_ElW5ISED4MhvR_YFYdmB/view?usp=drive_link): The dataset includes multi-region X-ray images focused on diagnosing bone fractures.
   - [Parking Lot Detection](https://drive.google.com/file/d/1Wehry7yNRMY5PELWkY6ysW_oQP44Xvzf/view?usp=sharing): The dataset is designed for developing and testing parking lot vehicle detection algorithms.
   
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand its structure, features, and the challenges it presents. Document your findings and initial thoughts on how to approach the classification problem.

3. **Model Development**:
   - Preprocess the images to get them ready for training.
   - Select and apply a machine learning algorithm to build a classifier. You can use frameworks like TensorFlow, Keras, or PyTorch.
   - Train your model and optimize its parameters to achieve the best performance.

4. **Evaluation**:
   - Validate your model using appropriate metrics (accuracy, precision, recall, F1-score, etc.).
   - Discuss the performance of your model and any potential biases or limitations.

5. **Deployment (Optional choices)**:
   - Deploy your model as a simple web application or a script that can take an image input and output a classification.
   - You can use streamlit + Flask for this, for example
   - Run your model on SageMaker

## Resources

- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Scikit-Learn for preprocessing tools: [https://scikit-learn.org/](https://scikit-learn.org/)

## Deliverables

1. **Python Code:** Provide well-documented Python code that conducts the analysis.
2. **Report**: Submit a detailed report documenting your EDA findings, model development process, evaluation metrics, and conclusions about the model's performance.
3. **Presentation**: Prepare a short presentation that overviews your project, from the dataset analysis to the final model evaluation. Include visual aids such as charts, model diagrams, and example predictions.

## Bonus

- Implement data augmentation techniques to improve your model's robustness.
- Compare the performance of at least three different models or architectures.
- Provide an interactive demo of your model during the presentation.

