# 🔥 PyTorch Workshop
A comprehensive PyTorch workshop covering the fundamentals and advanced techniques of deep learning.

## 📋 Table of Contents
1. **[Introduction to Tensors](./codes/00_tensor.ipynb)**  
   Learn about PyTorch tensors, the foundational data structure, and how to manipulate them.
1. **[Gradient and Autograd](./codes/01_gradient.ipynb)**  
   Understand gradients, automatic differentiation, and how PyTorch handles backpropagation with `autograd`.
1. **[Perceptron and Adaline](./codes/02_perceptron.ipynb)**  
   Explore the basics of the simplest neural network model (perceptron) and Adaptive Linear Neuron (Adaline).
1. **[Linear Regression](./codes/03_linear-regression.ipynb)**  
   Implement linear regression using PyTorch, including model training and prediction.
1. **[Activation Functions](./codes/04_activation-functions.ipynb)**  
   Study different activation functions (ReLU, Sigmoid, Tanh, ...) and their roles in neural networks.
1. **[Loss Functions](./codes/05_loss-functions.ipynb)**  
   Dive into common loss functions used in neural networks, including MSE, Cross-Entropy, and others.
1. **[Logistic Regression](./codes/06_logistic-regression.ipynb)**  
   Learn how to build a logistic regression model for binary classification tasks.
1. **[Working with Datasets](./codes/07_dataset.ipynb)**  
   Understand how to work with datasets in PyTorch using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.
1. **[Multi-Layer Perceptron](./codes/08_multi-layer-perceptron.ipynb)**  
   Implement and explore multi-layer perceptron (MLP) for more complex tasks.
1. **[Custom Classes in PyTorch](./codes/09_custom-class.ipynb)**  
   Learn how to define custom models, dataset, loss function, ... using PyTorch's class-based approach.
1. **[Radial Basis Function Networks](./codes/10_radial-basis-function-networks.ipynb)**  
   Implement and explore Radial Basis Function (RBF) networks and how they differ from traditional neural networks.
1. **[Transforms in PyTorch](./codes/11_transforms.ipynb)**  
   Learn to apply transforms like data augmentation on datasets using `torchvision.transforms.v2`.
1. **[Convolutional Neural Networks](./codes/12_convolutional-neural-networks.ipynb)**  
   Explore concepts around convolutional neural networks (CNNs).
1. **[Normalizations](./codes/13_normalizations.ipynb)**  
   Understand normalization techniques such as Batch Normalization and Layer Normalization.
1. **[Feature Extraction](./codes/14_feature-extraction.ipynb)**  
   Learn how to extract features from pre-trained models for downstream tasks.
1. **[Transfer Learning](./codes/15_transfer-learning.ipynb)**  
   Apply transfer learning by using pre-trained models for a new tasks.
1. **[Fine-Tuning Models](./codes/16_fine-tuning.ipynb)**  
   Understand how to fine-tune models by updating specific layers while freezing others.
1. **[Save and Load Checkpoints](./codes/17_save-load-checkpoint.ipynb)**  
   Learn how to save and load model checkpoints to resume training or for inference.

## 📦 Installing Dependencies
You can install all dependencies listed in `requirements.txt` using [pip](https://pip.pypa.io/en/stable/installation/).
```bash
pip install -r requirements.txt
```
**Note:** This project was developed using Python `v3.12.3`. If you encounter issues running the dependencies or code, consider using this specific Python version.

## 🛠️ Usage
   - Open the root folder with [VS Code](https://code.visualstudio.com/)
      - **Windows/Linux**: `Ctrl + K` followed by `Ctrl + O`
      - **macOS**: `Cmd + K` followed by `Cmd + O`
   - Open `.ipynb` files using [Jupyter](https://jupyter.org/) extension integrated with VS Code
   - Allow Visual Studio Code to install any recommended dependencies for working with Jupyter Notebooks.
   - Jupyter is integrated with both [VS Code](https://code.visualstudio.com/) & [Google Colab](https://colab.research.google.com/)

## 🔍 Find Me
Any mistakes, suggestions, or contributions? Feel free to reach out to me at:
   - 📍[linktr.ee/mr_pylin](https://linktr.ee/mr_pylin)
   
I look forward to connecting with you! 

## 📋 Prerequisites
   - Programming Fundamentals
      - Proficiency in [Python](https://github.com/mr-pylin/python-workshop) (data types, control structures, functions, etc.).
      - Experience with libraries like [NumPy](https://github.com/mr-pylin/numpy-workshop), Pandas and Matplotlib.
   - Mathematics for Machine Learning
      - Linear Algebra: Vectors, matrices, matrix operations.
         - [*Linear Algebra Review and Reference*](https://www.cs.cmu.edu/%7Ezkolter/course/linalg/linalg_notes.pdf) by [Zico Kolter](https://zicokolter.com)
         - [*Notes on Linear Algebra*](https://webspace.maths.qmul.ac.uk/p.j.cameron/notes/linalg.pdf) by [Peter J. Cameron](https://cameroncounts.github.io/web)
         - [*MATH 233 - Linear Algebra I Lecture Notes*](https://www.geneseo.edu/~aguilar/public/assets/courses/233/main_notes.pdf) by [Cesar O. Aguilar](https://www.geneseo.edu/~aguilar/)
      - Calculus: Derivatives, gradients, partial derivatives, chain rule (for backpropagation).
         - [*Lecture notes on advanced gradient descent*]() by [Cl´ement W. Royer](https://scholar.google.fr/citations?user=nmRlYWwAAAAJ&hl=en)
         - [*MATH 221 –  CALCULUS LECTURE NOTES VERSION 2.0*](https://people.math.wisc.edu/~angenent/Free-Lecture-Notes/free221.pdf) by [Sigurd Angenent](https://people.math.wisc.edu/~angenent)
         - [*Calculus*](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) by [Gilbert Strang](https://math.mit.edu/~gs)
      - Probability & Statistics: Probability distributions, mean/variance, etc.
         - [*MATH1024: Introduction to Probability and Statistics*](https://www.sujitsahu.com/teach/2020_math1024.pdf) by [Sujit Sahu](https://www.southampton.ac.uk/people/5wynjr/professor-sujit-sahu)

## 🔗 Usefull Links
   - **PyTorch**:
      - Source Code
         - Over 3000 contributers are currently working on PyTorch.
         - Link: [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
      - Website
         - The official website for PyTorch, offering comprehensive documentation, tutorials, and resources for deep learning and machine learning with PyTorch.
         - Link: [pytorch.org](https://pytorch.org/)
      - Pytorch Documentations
         - Detailed and comprehensive documentation for all PyTorch features and functionalities, including tutorials and guides to help you get started and master PyTorch.
         - Link: [pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
      - TorchVision Documentations:
         - The torchvision package [part of the PyTorch] consists of popular datasets, model architectures, and common image transformations for computer vision.
         - Link: [pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)
      - TorchAudio Documentation:
         - The torchaudio package [part of the PyTorch] consists of audio I/O and signal processing functionalities, enabling efficient loading, transforming, and manipulating audio.
         - Link: [pytorch.org/audio/stable/index.html](https://pytorch.org/audio/stable/index.html)
   - **NumPy**:
      - Website
         - The official website for NumPy, providing information, tutorials, and resources for the NumPy library
         - Link: [numpy.org](https://numpy.org/)
      - Documentation
         - Comprehensive guide and reference for all functionalities and features of the NumPy library
         - Link: [numpy.org/doc](https://numpy.org/doc/)
   - **MatPlotLib**:
      - A comprehensive library for creating static, animated, and interactive visualizations in Python
      - Link: [matplotlib.org](https://matplotlib.org/)
   - **Pandas**:
      - A powerful, open-source data analysis and manipulation library for Python
      - Pandas is built on top of NumPy
      - Link: [pandas.pydata.org/](https://pandas.pydata.org/)