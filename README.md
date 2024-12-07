# 🔥 PyTorch Workshop
A comprehensive PyTorch workshop covering the fundamentals and advanced techniques of deep learning.

## 📖 Table of Contents
### Main Notebooks
   1. [**Introduction to Tensors**](./codes/01_tensor.ipynb)  
   Learn about PyTorch tensors, the foundational data structure, and how to manipulate them.
   1. [**Gradient and Autograd**](./codes/02_gradient.ipynb)  
   Understand gradients, automatic differentiation, and how PyTorch handles backpropagation with `autograd`.
   1. [**Perceptron and AdaLiNe**](./codes/03_perceptron.ipynb)  
   Explore the basics of the simplest neural network model (perceptron) and Adaptive Linear Neuron (Adaline).
   1. [**Linear Regression**](./codes/04_linear-regression.ipynb)  
   Implement linear regression using PyTorch, including model training and prediction.
   1. [**Logistic Regression**](./codes/05_logistic-regression.ipynb)  
   Learn how to build a logistic regression model for binary classification tasks.
   1. [**Multi-Layer Perceptrons**](./codes/06_multi-layer-perceptrons.ipynb)  
   Implement and explore multi-layer perceptron (MLP) for more complex tasks.
   1. [**Radial Basis Function Networks**](./codes/07_radial-basis-function-networks.ipynb)  
   Implement and explore Radial Basis Function (RBF) networks and how they differ from traditional neural networks.
   1. [**Convolutional Neural Networks**](./codes/08_convolutional-neural-networks.ipynb)  
   Explore concepts around convolutional neural networks (CNNs).
   1. [**Feature Extraction**](./codes/09_feature-extraction.ipynb)  
   Learn how to extract features from pre-trained models for downstream tasks.
   1. [**Transfer Learning**](./codes/10_transfer-learning.ipynb)  
   Apply transfer learning by using pre-trained models for a new tasks.
   1. [**Fine-Tuning Models**](./codes/11_fine-tuning.ipynb)  
   Understand how to fine-tune models by updating specific layers while freezing others.
   1. [**Recurrent Neural Networks**](./codes/12_recurrent-neural-networks.ipynb)  
   Explore concepts around recurrent neural networks (RNNs).

### Utilities
A collection of concepts and tools utilized in the main notebooks for training models, ...  
   - [**Activation Functions**](./codes/utils/activation-functions.ipynb)  
      Study different activation functions (ReLU, Sigmoid, Tanh, ...) and their roles in neural networks.
   - [**Checkpoints**](./codes/utils/checkpoints.ipynb)  
      Learn how to save and load model checkpoints to resume training or for inference.
   - [**Custom Implementations**](./codes/utils/customs.ipynb)  
      Learn how to define custom models, dataset, loss function, ... using PyTorch's class-based approach.
   - [**Working with Datasets**](./codes/utils/dataset-dataloader.ipynb)  
      Understand how to work with datasets in PyTorch using `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.
   - [**Parameters vs. Hyperparameters**](./codes/utils/hyperparameters.ipynb)  
      Understand the difference between parameters and hyperparameters in neural networks.
   - [**Loss Functions**](./codes/utils/loss-functions.ipynb)  
      Dive into common loss functions used in neural networks, including MSE, Cross-Entropy, and others.
   - [**Normalizations**](./codes/utils/normalizations.ipynb)  
      Understand normalization techniques such as Batch Normalization and Layer Normalization.
   - [**Data Transforms**](./codes/utils/transforms.ipynb)  
      Learn to apply transforms like data augmentation on datasets using `torchvision.transforms.v2`.

### Models
- **CNN Architectures**
   1. [**LeNet-5 Architecture**](./codes/models/CNN/01_lenet5-architecture.ipynb)
   1. [**AlexNet Architecture**](./codes/models/CNN/02_alexnet-architecture.ipynb)
   1. [**VGGNet Architecture**](./codes/models/CNN/03_vggnet-architecture.ipynb)
   1. [**GoogLeNet Architecture**](./codes/models/CNN/04_googlenet-architecture.ipynb)
   1. [**ResNet Architecture**](./codes/models/CNN/05_resnet-architecture.ipynb)
   1. [**DenseNet Architecture**](./codes/models/CNN/06_densenet-architecture.ipynb)
   1. [**EfficientNet Architecture**](./codes/models/CNN/07_efficientnet-architecture.ipynb)

### Projects
   1. [**Multi-Layer Perceptrons**](./codes/projects/01_multi-layer-perceptrons.ipynb)  
   A comprehensive example to building, training, and deploying a MLP, from data preprocessing to prediction.
   1. [**Convolutional Neural Networks**](./codes/projects/02_convolutional-neural-networks.ipynb)  
   A comprehensive example to building, training, and deploying a CNN, from data preprocessing to prediction.


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
This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to use, modify, and distribute this code, but you must include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.

## ✍️ Additional Licensing Information
- **Original Images**:
   - The images located in the [./assets/images/original/](./assets/images/original/) folder are licensed under the **[CC BY-ND 4.0](./assets/images/original/LICENSE)**.
   - Note: This license restricts derivative works, meaning you may share these images but cannot modify them.

- **Third-Party Assets**:
   - Additional images located in [./assets/images/third_party/](./assets/images/third_party/) are used with permission or according to their original licenses.
   - Attributions and references to original sources are included in the code where these images are used.