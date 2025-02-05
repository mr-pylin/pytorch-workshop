# 🔥 PyTorch Workshop

[![License](https://img.shields.io/github/license/mr-pylin/pytorch-workshop?color=blue)](https://github.com/mr-pylin/pytorch-workshop/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.12.8-yellow?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3128/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1faf9d4577d3406a9ac65a4cb8d3d4f1)](https://app.codacy.com/gh/mr-pylin/pytorch-workshop/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![Repo Size](https://img.shields.io/github/repo-size/mr-pylin/pytorch-workshop?color=lightblue)
![Last Updated](https://img.shields.io/github/last-commit/mr-pylin/pytorch-workshop?color=orange)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?color=brightgreen)](https://github.com/mr-pylin/pytorch-workshop/pulls)

A comprehensive **PyTorch** workshop covering the fundamentals and advanced techniques of deep learning.

## 📖 Table of Contents

### 📖 Main Notebooks

1. [**Introduction to Tensors**](./code/01-tensor.ipynb)  
Learn about PyTorch tensors, the foundational data structure, and how to manipulate them.
1. [**Gradient and Autograd**](./code/02-gradient.ipynb)  
Understand gradients, automatic differentiation, and how PyTorch handles backpropagation with `autograd`.
1. [**Perceptron and AdaLiNe**](./code/03-simple-neurons.ipynb)  
Explore the basics of the simplest neural network model (perceptron) and Adaptive Linear Neuron (Adaline).
1. [**Regression Models**](./code/04-regression-model.ipynb)  
Implement linear and logistic regression using PyTorch, including model training and prediction.
1. [**Multi-Layer Perceptrons**](./code/05-multi-layer-perceptrons.ipynb)  
Implement and explore multi-layer perceptron (MLP) for more complex tasks.
1. [**Radial Basis Function Networks**](./code/06-radial-basis-function-networks.ipynb)  
Implement and explore Radial Basis Function (RBF) networks and how they differ from traditional neural networks.
1. [**Convolutional Neural Networks**](./code/07-convolutional-neural-networks.ipynb)  
Explore concepts around convolutional neural networks (CNNs).
1. [**Feature Extraction**](./code/08-feature-extraction.ipynb)  
Learn how to extract features from pre-trained models for downstream tasks.
1. [**Transfer Learning**](./code/09-transfer-learning.ipynb)  
Apply transfer learning by using pre-trained models for a new tasks.
1. [**Fine-Tuning Models**](./code/10-fine-tuning.ipynb)  
Understand how to fine-tune models by updating specific layers while freezing others.
1. [**Recurrent Neural Networks**](./code/11-recurrent-neural-networks.ipynb)  
Explore concepts around recurrent neural networks (RNNs).

### 📖 Utilities

A collection of concepts and tools utilized in the main notebooks for training models, ...  

- [**Activation Functions**](./code/utils/activation.ipynb)  
Study different activation functions (ReLU, Sigmoid, Tanh, ...) and their roles in neural networks.
- [**Checkpoints**](./code/utils/checkpoint.ipynb)  
Learn how to save and load model checkpoints to resume training or for inference.
- [**Working with Datasets**](./code/utils/dataset.ipynb)  
Understand how to work with datasets in PyTorch using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.
- [**Parameters vs. Hyperparameters**](./code/utils/hyperparameter.ipynb)  
Understand the difference between parameters and hyperparameters in neural networks.
- [**Loss Functions**](./code/utils/loss.ipynb)  
Dive into common loss functions used in neural networks, including MSE, Cross-Entropy, and others.
- [**Metrics**](./code/utils/metric.ipynb)  
Learn about evaluation metrics such as Accuracy, Precision, Recall, F1-Score, and custom metric implementations.
- [**Model Creation**](./code/utils/model-creation.ipynb)  
Study how to initialize built-in models architectures or create custom sequential/non-sequential models.
- [**Normalization Techniques**](./code/utils/normalization.ipynb)  
Understand normalization techniques such as Batch Normalization and Layer Normalization.
- [**Optimizers**](./code/utils/optimizer.ipynb)  
Explore different built-in optimizers or how to create custom optimizers.
- [**Vision Transforms**](./code/utils/vision-transform.ipynb)  
Learn to apply transforms like data augmentation on datasets using `torchvision.transforms.v2`.
- [**Word Embeddings**](./code/utils/word-embedding.ipynb)  
Explore different word embedding techniques and their applications in natural language processing.

### 📖 Models

- **CNN Architectures**
  1. [**LeNet-5 Architecture**](./code/models/cnn/lenet5-architecture.ipynb)
  1. [**AlexNet Architecture**](./code/models/cnn/alexnet-architecture.ipynb)
  1. [**VGGNet Architecture**](./code/models/cnn/vggnet-architecture.ipynb)
  1. [**GoogLeNet Architecture**](./code/models/cnn/googlenet-architecture.ipynb)
  1. [**Xception Architecture**](./code/models/cnn/xception-architecture.ipynb)
  1. [**ResNet Architecture**](./code/models/cnn/resnet-architecture.ipynb)
  1. [**DenseNet Architecture**](./code/models/cnn/densenet-architecture.ipynb)
  1. [**EfficientNet Architecture**](./code/models/cnn/efficientnet-architecture.ipynb)

### 📖 Projects

Implementation details are provided in the **README** files within the parent directories.

1. [**MNIST Classification**](./code/projects/mnist-classification/)  
    - [**Implementation 1**](./code/projects/mnist-classification/implementation-1/)
1. [**CIFAR-10 Classification**](./code/projects/cifar-classification/cifar-10/)  
    - [**Implementation 1**](./code/projects/cifar-classification/cifar-10/implementation-1/)

## 📋 Prerequisites

- 👨‍💻 **Programming Fundamentals**
  - Proficiency in **Python** (data types, control structures, functions, classes, etc.).
    - My Python Workshop: [github.com/mr-pylin/python-workshop](https://github.com/mr-pylin/python-workshop)
  - Experience with libraries like **NumPy**, **Pandas** and **Matplotlib**.
    - My NumPy Workshop: [github.com/mr-pylin/numpy-workshop](https://github.com/mr-pylin/numpy-workshop)
    - My Pandas Workshop: [Coming Soon](https://github.com/mr-pylin/#)
    - My Data Visualization Workshop: [github.com/mr-pylin/data-visualization-workshop](https://github.com/mr-pylin/data-visualization-workshop)
- 🔣 **Mathematics for Machine Learning**
  - 🔲 **Linear Algebra**: Vectors, matrices, matrix operations.
    - [**Linear Algebra Review and Reference**](https://www.cs.cmu.edu/%7Ezkolter/course/linalg/linalg_notes.pdf) written by [*Zico Kolter*](https://zicokolter.com).
    - [**Notes on Linear Algebra**](https://webspace.maths.qmul.ac.uk/p.j.cameron/notes/linalg.pdf) written by [*Peter J. Cameron*](https://cameroncounts.github.io/web).
    - [**MATH 233 - Linear Algebra I Lecture Notes**](https://www.geneseo.edu/~aguilar/public/assets/courses/233/main_notes.pdf) written by [*Cesar O. Aguilar*](https://www.geneseo.edu/~aguilar/).
  - 📈 **Calculus**: Derivatives, gradients, partial derivatives, chain rule (for backpropagation).
    - [**Lecture notes on advanced gradient descent**](https://www.lamsade.dauphine.fr/~croyer/ensdocs/GD/LectureNotesOML-GD.pdf) written by [*Cl´ement W. Royer*](https://scholar.google.fr/citations?user=nmRlYWwAAAAJ&hl=en).
    - [**MATH 221 –  CALCULUS LECTURE NOTES VERSION 2.0**](https://people.math.wisc.edu/~angenent/Free-Lecture-Notes/free221.pdf) written by [*Sigurd Angenent*](https://people.math.wisc.edu/~angenent).
    - [**Calculus**](https://ocw.mit.edu/ans7870/resources/Strang/Edited/Calculus/Calculus.pdf) written by [*Gilbert Strang*](https://math.mit.edu/~gs).
  - 🎲 **Probability & Statistics**: Probability distributions, mean/variance, etc.
    - [**MATH1024: Introduction to Probability and Statistics**](https://www.sujitsahu.com/teach/2020_math1024.pdf) written by [*Sujit Sahu*](https://www.southampton.ac.uk/people/5wynjr/professor-sujit-sahu).

## ⚙️ Setup

This project requires Python **v3.10** or higher. It was developed and tested using Python **v3.12.8**. If you encounter issues running the specified version of dependencies, consider using this version of Python.

### 📝 List of Dependencies

[![datasets](https://img.shields.io/badge/datasets-3.2.0-purple)](https://pypi.org/project/datasets/3.2.0/)
[![ipykernel](https://img.shields.io/badge/ipykernel-6.29.5-ff69b4)](https://pypi.org/project/ipykernel/6.29.5/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-8.1.5-ff6347)](https://pypi.org/project/ipywidgets/8.1.5/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.0-green)](https://pypi.org/project/matplotlib/3.10.0/)
[![numpy](https://img.shields.io/badge/numpy-2.2.1-orange)](https://pypi.org/project/numpy/2.2.1/)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-yellow)](https://pypi.org/project/pandas/2.2.3/)
[![PySoundFile](https://img.shields.io/badge/PySoundFile-0.9.0.post1-red)](https://pypi.org/project/PySoundFile/0.9.0.post1/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.0-darkblue)](https://pypi.org/project/scikit-learn/1.6.0/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13.2-lightblue)](https://pypi.org/project/seaborn/0.13.2/)
[![torch](https://img.shields.io/badge/torch-2.5.1%2Bcu124-gold)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.5.1%2Bcu124-lightgreen)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.20.1%2Bcu124-teal)](https://pytorch.org/)
[![torchinfo](https://img.shields.io/badge/torchinfo-1.8.0-blueviolet)](https://pypi.org/project/torchinfo/1.8.0/)
[![torchmetrics](https://img.shields.io/badge/torchmetrics-1.6.1-lightgray)](https://pypi.org/project/torchmetrics/1.6.1/)

### 📦 Install Dependencies

#### 📦 Method 1: Poetry (Recommended)

Use [**Poetry**](https://python-poetry.org/) for dependency management. It handles dependencies, virtual environments, and locking versions more efficiently than pip. To install dependencies using Poetry:

- **Option 1 [Recommended]**: Install exact dependency versions specified in [**poetry.lock**](./poetry.lock) for consistent environments:

  ```bash
  poetry install
  ```

- **Option 2**: Install the latest compatible dependency versions from [**pyproject.toml**](./pyproject.toml) and regenerate the [**poetry.lock**](./poetry.lock) file:

  ```bash
  poetry install --no-root
  ```

#### 📦 Method 2: Pip

Install all dependencies listed in [**requirements.txt**](./requirements.txt) using [**pip**](https://pip.pypa.io/en/stable/installation/):

```bash
pip install -r requirements.txt
```

#### 🌐 Connection Issues

If you encounter connection issues during installation, you can try extending the **timeout** and increasing the number of **retries** with the following:

- **For Poetry**: Use the following command to set the retries and timeout directly in the terminal **before running the install**:
  - **Windows**:
    - **PowerShell**:

      ```bash
      $env:POETRY_HTTP_TIMEOUT=300
      $env:POETRY_HTTP_RETRIES=10
      ```

    - **Command Prompt**:

      ```bash
      set POETRY_HTTP_TIMEOUT=300
      set POETRY_HTTP_RETRIES=10
      ```

  - **Linux/macOS**:
    - **Terminal**:

      ```bash
      export POETRY_HTTP_TIMEOUT=300
      export POETRY_HTTP_RETRIES=10
      ```

- **For Pip**: Use the `--retries` and `--timeout` flags directly in your pip command:

  ```bash
  pip install -r requirements.txt --retries 10 --timeout 300
  ```

### 🛠️ Usage Instructions

1. Open the root folder with [**VS Code**](https://code.visualstudio.com/) (`Ctrl/Cmd + K` followed by `Ctrl/Cmd + O`).
1. Open `.ipynb` files using the [**Jupyter extension**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) integrated with **VS Code**.
1. Select the correct Python kernel and virtual environment where the dependencies were installed.
1. Allow **VS Code** to install any recommended dependencies for working with Jupyter Notebooks.

✍️ **Notes**:  

- It is **highly recommended** to stick with the exact dependency versions specified in [**poetry.lock**](./poetry.lock) or [**requirements.txt**](./requirements.txt) rather than using the latest package versions. The repository has been **tested** on these versions to ensure **compatibility** and **stability**.
- This repository is **actively maintained**, and dependencies are **updated regularly** to the latest **stable** versions.
- The **table of contents** embedded in the **notebooks** may not function correctly on **GitHub**.
- For an improved experience, open the notebooks **locally** or view them via [**nbviewer**](https://nbviewer.org/github/mr-pylin/pytorch-workshop).

## 🔗 Useful Links

### **PyTorch**

- **Source Code**:
  - Over **3500** contributors are currently working on PyTorch.
  - *Link*: [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- **Website**:
  - The **official** website for PyTorch, offering comprehensive documentation, tutorials, and resources for deep learning and machine learning with PyTorch.
  - Link: [pytorch.org](https://pytorch.org/)
- **Pytorch Documentations**:
  - Detailed and comprehensive documentation for all PyTorch features and functionalities, including tutorials and guides to help you get started and master PyTorch.
  - Link: [pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- **TorchVision Documentations**:
  - The torchvision package [part of the PyTorch] consists of popular datasets, model architectures, and common image transformations for computer vision.
  - Link: [pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)
- **TorchAudio Documentation**:
  - The torchaudio package [part of the PyTorch] consists of audio I/O and signal processing functionalities, enabling efficient loading, transforming, and manipulating audio.
  - Link: [pytorch.org/audio/stable/index.html](https://pytorch.org/audio/stable/index.html)

### **NumPy**

- A fundamental package for scientific computing in Python, providing support for **arrays**, **matrices**, and a large collection of **mathematical functions**.
- Official site: [numpy.org](https://numpy.org/)

### **Pandas**

- A powerful, open-source data analysis and manipulation library for Python.
- Pandas is built on top of NumPy.
- Official site: [pandas.pydata.org](https://pandas.pydata.org/)

### **Data Visualization**

- A comprehensive collection of Python libraries for creating static, animated, and interactive visualizations: **Matplotlib**, **Seaborn**, and **Plotly**.
- Official sites: [matplotlib.org](https://matplotlib.org/) | [seaborn.pydata.org](https://seaborn.pydata.org/) | [plotly.com](https://plotly.com/)

## 🔍 Find Me

Any mistakes, suggestions, or contributions? Feel free to reach out to me at:

- 📍[**linktr.ee/mr_pylin**](https://linktr.ee/mr_pylin)

I look forward to connecting with you! 🏃‍♂️

## 📄 License

This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to **use**, **modify**, and **distribute** this code, but you **must** include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.

### ✍️ Additional Licensing Information

- **Original Images**:
  - The images located in the [./assets/images/original/](./assets/images/original/) folder are licensed under the **[CC BY-ND 4.0](./assets/images/original/LICENSE)**.
  - Note: This license restricts derivative works, meaning you may share these images but cannot modify them.
- **Third-Party Assets**:
  - Additional images located in [./assets/images/third_party/](./assets/images/third_party/) are used with permission or according to their original licenses.
  - Attributions and references to original sources are included in the code where these images are used.
