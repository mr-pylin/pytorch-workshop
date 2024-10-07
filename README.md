# 🔥 PyTorch Workshop
A comprehensive PyTorch workshop covering the fundamentals and advanced techniques of deep learning.

## 📖 Table of Contents
0. **[Introduction to Tensors](./codes/00_tensor.ipynb)**  
   Learn about PyTorch tensors, the foundational data structure, and how to manipulate them.
0. **[Gradient and Autograd](./codes/01_gradient.ipynb)**  
   Understand gradients, automatic differentiation, and how PyTorch handles backpropagation with `autograd`.
0. **[Perceptron and Adaline](./codes/02_perceptron.ipynb)**  
   Explore the basics of the simplest neural network model (perceptron) and Adaptive Linear Neuron (Adaline).
0. **[Linear Regression](./codes/03_linear-regression.ipynb)**  
   Implement linear regression using PyTorch, including model training and prediction.
0. **[Activation Functions](./codes/04_activation-functions.ipynb)**  
   Study different activation functions (ReLU, Sigmoid, Tanh, ...) and their roles in neural networks.
0. **[Loss Functions](./codes/05_loss-functions.ipynb)**  
   Dive into common loss functions used in neural networks, including MSE, Cross-Entropy, and others.
0. **[Logistic Regression](./codes/06_logistic-regression.ipynb)**  
   Learn how to build a logistic regression model for binary classification tasks.
0. **[Working with Datasets](./codes/07_dataset.ipynb)**  
   Understand how to work with datasets in PyTorch using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.
0. **[Multi-Layer Perceptron](./codes/08_multi-layer-perceptron.ipynb)**  
   Implement and explore multi-layer perceptron (MLP) for more complex tasks.
0. **[Custom Classes in PyTorch](./codes/09_custom-class.ipynb)**  
   Learn how to define custom models, dataset, loss function, ... using PyTorch's class-based approach.
0. **[Radial Basis Function Networks](./codes/10_radial-basis-function-networks.ipynb)**  
   Implement and explore Radial Basis Function (RBF) networks and how they differ from traditional neural networks.
0. **[Transforms in PyTorch](./codes/11_transforms.ipynb)**  
   Learn to apply transforms like data augmentation on datasets using `torchvision.transforms.v2`.
0. **[Convolutional Neural Networks](./codes/12_convolutional-neural-networks.ipynb)**  
   Explore concepts around convolutional neural networks (CNNs).
0. **[Normalizations](./codes/13_normalizations.ipynb)**  
   Understand normalization techniques such as Batch Normalization and Layer Normalization.
0. **[Feature Extraction](./codes/14_feature-extraction.ipynb)**  
   Learn how to extract features from pre-trained models for downstream tasks.
0. **[Transfer Learning](./codes/15_transfer-learning.ipynb)**  
   Apply transfer learning by using pre-trained models for a new tasks.
0. **[Fine-Tuning Models](./codes/16_fine-tuning.ipynb)**  
   Understand how to fine-tune models by updating specific layers while freezing others.
0. **[Save and Load Checkpoints](./codes/17_save-load-checkpoint.ipynb)**  
   Learn how to save and load model checkpoints to resume training or for inference.
0. **[Recurrent Neural Networks](./codes/18_recurrent-neural-network.ipynb)**  
   Explore concepts around recurrent neural networks (RNNs).

## 📋 Prerequisites
   - **Programming Fundamentals**
      - Proficiency in Python (data types, control structures, functions, etc.).
         - My Python Workshop: [github.com/mr-pylin/python-workshop](https://github.com/mr-pylin/python-workshop)
      - Experience with libraries like **NumPy**, **Pandas** and **Matplotlib**.
         - My NumPy Workshop: [github.com/mr-pylin/numpy-workshop](https://github.com/mr-pylin/numpy-workshop)
         - My Pandas Workshop: [Coming Soon](https://github.com/mr-pylin/#)
         - My MatPlotLib Workshop: [Coming Soon](https://github.com/mr-pylin/#)
   - **Mathematics for Machine Learning**
      - Linear Algebra: Vectors, matrices, matrix operations.
         - [*Linear Algebra Review and Reference*](https://www.cs.cmu.edu/%7Ezkolter/course/linalg/linalg_notes.pdf) written by [Zico Kolter](https://zicokolter.com)
         - [*Notes on Linear Algebra*](https://webspace.maths.qmul.ac.uk/p.j.cameron/notes/linalg.pdf) written by [Peter J. Cameron](https://cameroncounts.github.io/web)
         - [*MATH 233 - Linear Algebra I Lecture Notes*](https://www.geneseo.edu/~aguilar/public/assets/courses/233/main_notes.pdf) written by [Cesar O. Aguilar](https://www.geneseo.edu/~aguilar/)
      - Calculus: Derivatives, gradients, partial derivatives, chain rule (for backpropagation).
         - [*Lecture notes on advanced gradient descent*](https://www.lamsade.dauphine.fr/~croyer/ensdocs/GD/LectureNotesOML-GD.pdf) written by [Cl´ement W. Royer](https://scholar.google.fr/citations?user=nmRlYWwAAAAJ&hl=en)
         - [*MATH 221 –  CALCULUS LECTURE NOTES VERSION 2.0*](https://people.math.wisc.edu/~angenent/Free-Lecture-Notes/free221.pdf) written by [Sigurd Angenent](https://people.math.wisc.edu/~angenent)
         - [*Calculus*](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) written by [Gilbert Strang](https://math.mit.edu/~gs)
      - Probability & Statistics: Probability distributions, mean/variance, etc.
         - [*MATH1024: Introduction to Probability and Statistics*](https://www.sujitsahu.com/teach/2020_math1024.pdf) written by [Sujit Sahu](https://www.southampton.ac.uk/people/5wynjr/professor-sujit-sahu)

# ⚙️ Setup
This project was developed using Python `v3.12.3`. If you encounter issues running the specified version of dependencies, consider using this specific Python version.

## 📦 Installing Dependencies
You can install all dependencies listed in `requirements.txt` using [pip](https://pip.pypa.io/en/stable/installation/).
```bash
pip install -r requirements.txt
```

## 🛠️ Usage Instructions
   - Open the root folder with [VS Code](https://code.visualstudio.com/)
      - **Windows/Linux**: `Ctrl + K` followed by `Ctrl + O`
      - **macOS**: `Cmd + K` followed by `Cmd + O`
   - Open `.ipynb` files using [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) integrated with **VS Code**
   - Allow **VS Code** to install any recommended dependencies for working with Jupyter Notebooks.
   - Note: Jupyter is integrated with both **VS Code** & **[Google Colab](https://colab.research.google.com/)**

# 🔗 Useful Links
   - **PyTorch**:
      - Source Code
         - Over 3500 contributors are currently working on PyTorch.
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
   - **NumPy**
      - A fundamental package for scientific computing in Python, providing support for arrays, matrices, and a large collection of mathematical functions.
      - Official site: [numpy.org](https://numpy.org/)
   - **Pandas**:
      - A powerful, open-source data analysis and manipulation library for Python
      - Pandas is built on top of NumPy
      - Official site: [pandas.pydata.org](https://pandas.pydata.org/)
   - **MatPlotLib**:
      - A comprehensive library for creating static, animated, and interactive visualizations in Python
      - Official site: [matplotlib.org](https://matplotlib.org/)

# 🔍 Find Me
Any mistakes, suggestions, or contributions? Feel free to reach out to me at:
   - 📍[linktr.ee/mr_pylin](https://linktr.ee/mr_pylin)
   
I look forward to connecting with you! 🏃‍♂️

# 📄 License
This repository is licensed under the **[MIT License](./LICENSE)**, except for the contents in the [./assets/images/SVGs/](./assets/images/SVGs/) path, which are licensed under the **[CC BY-ND 4.0](./assets/images/SVGs/LICENSE)**.