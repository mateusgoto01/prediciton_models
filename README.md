<p align="center">
  <a href="" rel="noopener">
  </a>
</p>

<h3 align="center">Introduction to Machine Learning Project</h3>

<div align="center">
  <a href=""><img src="https://img.shields.io/badge/status-active-success.svg" alt="Status"></a>
  <a href="https://github.com/yourusername/your-repo-name/issues"><img src="https://img.shields.io/github/issues/yourusername/your-repo-name.svg" alt="GitHub Issues"></a>
  <a href="https://github.com/yourusername/your-repo-name/pulls"><img src="https://img.shields.io/github/issues-pr/yourusername/your-repo-name.svg" alt="GitHub Pull Requests"></a>
  <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</div>

---

<p align="center">
  This project is an introduction to machine learning, focusing on implementing and training neural networks for various applications.
  <br> 
</p>

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Installation](#installation)
- [Showcase](#showcase)
  - [Perceptron Training](#q1)
  - [Digit Classification](#q2)
  - [Regression Task](#q3)
  - [Language Identification](#q4)

## 🧐 About <a name = "about"></a>
This project serves as an introduction to machine learning. It includes the implementation and training of Perceptron and neural network models for various tasks such as regression, digit classification, and language identification.

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will help you set up the project on your local machine for development and testing purposes.

## 🔧 Installation <a name = "installation"></a>
To run this project, you will need to install the following libraries:
- `numpy`: for support with large multi-dimensional arrays
- `matplotlib`: a 2D plotting library

To install these libraries, use the following commands:
```bash
pip install numpy
pip install matplotlib
```
## 🌟 Showcase <a name="showcase"></a>
This section showcases the solutions to each problem:
## Perceptron Training <a name="q1"></a>

Description: Implement and train a Perceptron model to classify linearly separable data. The Perceptron algorithm adjusts weights based on misclassifications during training, aiming to find a decision boundary that separates different classes.

Running the Code: To train and test the Perceptron model, run:
```bas
python autograder.py -q q1
```
You should see the results like:

![Q1](https://i.imgur.com/MLzgqPE.png)

## Non-linear Regression <a name="q2"></a>

In this question, the task is to train a neural network to approximate the sine function \( \sin(x) \) over the interval \([-2\pi, 2\pi]\). The goal is to implement a `RegressionModel` class in Python, specifically completing the methods `__init__`, `run`, `get_loss`, and `train`.

**Task Overview:**
1. **Initialization:** Implement the `__init__` method of the `RegressionModel` class to initialize the neural network model.
2. **Model Prediction:** Implement the `run` method to return predictions of the model for given input data.
3. **Loss Calculation:** Implement the `get_loss` method to compute the loss between model predictions and target outputs.
```bas
python autograder.py -q q2
```
![Q2](https://i.imgur.com/UkJuV0j.png)

## Digit Classification <a name="q3"></a>
Description: Implement a neural network to classify handwritten digits using the MNIST dataset. The neural network uses forward and backward propagation to adjust weights and minimize classification error. The model aims to accurately identify digits from 0 to 9 based on pixel values.

Running the Code: To train and test the digit classification model, run:
```bas
python autograder.py -q q3
```

![Q3](https://i.imgur.com/tuYqHJR.png)
## Language Identification <a name="q4"></a>
Description: Create a neural network model to identify languages from text samples. The model processes textual data and uses features such as character or word frequencies to determine the language of the input text. The aim is to accurately classify the language from a set of predefined options.

Running the Code: To train and test the language identification model, run:
```bas
python autograder.py -q q4
```
Example of output:
| Word       | True Language | Confidence | Predicted Language |
|------------|---------------|------------|--------------------|
| cuz        | English       | 4.9%       | Polish             |
| endure     | English       | 14.7%      | Spanish            |
| prophecy   | English       | 81.4%      |                    |
| octubre    | Spanish       | 84.3%      |                    |
| salvó      | Spanish       | 98.0%      |                    |
| opina      | Spanish       | 28.3%      | Finnish            |
| pelastit   | Finnish       | 64.8%      |                    |
| valitsit   | Finnish       | 89.3%      |                    |
| lehdessä   | Finnish       | 99.7%      |                    |
| opgelucht  | Dutch         | 90.1%      |                    |
| belang     | Dutch         | 64.3%      |                    |
| richt      | Dutch         | 57.0%      |                    |
| firmy      | Polish        | 36.4%      | English            |
| impreza    | Polish        | 6.8%       | Spanish            |
| telewizję  | Polish        | 97.6%      |                    |
