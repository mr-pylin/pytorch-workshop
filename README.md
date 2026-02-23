# 🔥 PyTorch Workshop

[![License](https://img.shields.io/github/license/mr-pylin/pytorch-workshop?color=blue)](https://github.com/mr-pylin/pytorch-workshop/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.13.9-yellow?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3139/)
[![torch](https://img.shields.io/badge/torch-2.9.0-gold?logo=pytorch)](https://pytorch.org/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1faf9d4577d3406a9ac65a4cb8d3d4f1)](https://app.codacy.com/gh/mr-pylin/pytorch-workshop/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![Repo Size](https://img.shields.io/github/repo-size/mr-pylin/pytorch-workshop?color=lightblue)
![Last Updated](https://img.shields.io/github/last-commit/mr-pylin/pytorch-workshop?color=orange)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?color=brightgreen)](https://github.com/mr-pylin/pytorch-workshop/pulls)
[![Telegram](https://img.shields.io/badge/Telegram-Group-black?logo=telegram)](https://t.me/python_topics)

A comprehensive **PyTorch** workshop covering the fundamentals and advanced techniques of deep learning.

## 📖 Table of Contents

### 📘 Essentials

1. [**Linear Algebra Basics**](./code/base/01-linalg-basics.ipynb)
1. [**Introductions to Convolution**](./code/base/02-convolution.ipynb)

### 📖 Main Notebooks

1. [**Introduction to Tensors**](./code/01-tensor.ipynb)
1. [**Gradient and Autograd**](./code/02-gradient.ipynb)
1. [**Perceptron and AdaLiNe**](./code/03-simple-neurons.ipynb)
1. [**Regression Models**](./code/04-regression-model.ipynb)
1. [**Multi-Layer Perceptrons**](./code/05-multi-layer-perceptrons.ipynb)
1. [**Radial Basis Function Networks**](./code/06-radial-basis-function-networks.ipynb)
1. [**Convolutional Neural Networks**](./code/07-convolutional-neural-networks.ipynb)
1. [**Feature Extraction**](./code/08-feature-extraction.ipynb)
1. [**Transfer Learning**](./code/09-transfer-learning.ipynb)
1. [**Fine-Tuning Models**](./code/10-fine-tuning.ipynb)
1. [**Recurrent Neural Networks**](./code/11-recurrent-neural-networks.ipynb)

### 📖 Utilities

A collection of concepts and tools utilized in the main notebooks for training models, ...  

- [**Activation Functions**](./code/utils/activation.ipynb)
- [**Checkpoints**](./code/utils/checkpoint.ipynb)
- [**Working with Datasets**](./code/utils/dataset.ipynb)
- [**Parameters vs. Hyperparameters**](./code/utils/hyperparameter.ipynb)
- [**Loss Functions**](./code/utils/loss.ipynb)
- [**Metrics**](./code/utils/metric.ipynb)
- [**Model Creation**](./code/utils/model-creation.ipynb)
- [**Normalization Techniques**](./code/utils/normalization.ipynb)
- [**Optimizers**](./code/utils/optimizer.ipynb)
- [**Vision Transforms**](./code/utils/vision-transform.ipynb)
- [**Word Embeddings**](./code/utils/word-embedding.ipynb)

### 📖 Models

- **CNN Architectures**
  1. [**Classic and Foundational**](./code/models/cnn/classic-and-foundational.ipynb)
     - **LeNet-5**
     - **AlexNet**
  1. [**VGGNet Architecture**](./code/models/cnn/vggnet.ipynb)
  1. [**GoogLeNet Architecture**](./code/models/cnn/googlenet.ipynb)
  1. [**Xception Architecture**](./code/models/cnn/xception.ipynb)
  1. [**ResNet Architecture**](./code/models/cnn/resnet.ipynb)
  1. [**DenseNet Architecture**](./code/models/cnn/densenet.ipynb)
  1. [**EfficientNet Architecture**](./code/models/cnn/efficientnet.ipynb)

### 📖 Projects

Implementation details are provided in the **README** files within the parent directories.

1. [**MNIST Classification**](./code/projects/mnist-classification/)  
    - [**Implementation 1**](./code/projects/mnist-classification/implementation-1/)
1. [**CIFAR-10 Classification**](./code/projects/cifar-classification/cifar-10/)  
    - [**Implementation 1**](./code/projects/cifar-classification/cifar-10/implementation-1/)

## 📋 Prerequisites

> [!NOTE]
> The mathematics requirements below are **not mandatory** but will significantly deepen your understanding of machine learning concepts.

- 👨‍💻 **Programming Fundamentals**
  - Proficiency in **Python** (data types, control structures, functions, classes, etc.).
    - My Python Workshop: [**github.com/mr-pylin/python-workshop**](https://github.com/mr-pylin/python-workshop)
  - Experience with libraries like **NumPy**, **Pandas** and **Matplotlib**.
    - My NumPy Workshop: [**github.com/mr-pylin/numpy-workshop**](https://github.com/mr-pylin/numpy-workshop)
    - My Pandas Workshop: [**github.com/mr-pylin/pandas-workshop**](https://github.com/mr-pylin/pandas-workshop)
    - My Data Visualization Workshop: [**github.com/mr-pylin/data-visualization-workshop**](https://github.com/mr-pylin/data-visualization-workshop)
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

> [!IMPORTANT]
> This project requires Python **v3.10** or higher. It was developed and tested using Python **v3.13.9**. If you encounter issues running the specified version of dependencies, consider using this version of Python.

### 📝 List of Dependencies

[![datasets](https://img.shields.io/badge/datasets-4.3.0-purple)](https://pypi.org/project/datasets/4.3.0/)
[![ipykernel](https://img.shields.io/badge/ipykernel-7.0.1-ff69b4)](https://pypi.org/project/ipykernel/7.0.1/)
[![ipympl](https://img.shields.io/badge/ipympl-0.9.8-orange)](https://pypi.org/project/ipympl/0.9.8/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-8.1.7-red)](https://pypi.org/project/ipywidgets/8.1.7/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.7-green)](https://pypi.org/project/matplotlib/3.10.7/)
[![numpy](https://img.shields.io/badge/numpy-2.3.4-orange)](https://pypi.org/project/numpy/2.3.4/)
[![opencv-python](https://img.shields.io/badge/opencv--python-4.11.0.86-blue)](https://pypi.org/project/opencv-python/4.11.0.86/)
[![pandas](https://img.shields.io/badge/pandas-2.3.3-yellow)](https://pypi.org/project/pandas/2.3.3/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-darkblue)](https://pypi.org/project/scikit-learn/1.7.2/)
[![seaborn](https://img.shields.io/badge/seaborn-0.13.2-lightblue)](https://pypi.org/project/seaborn/0.13.2/)
[![torch](https://img.shields.io/badge/torch-2.9.0-gold)](https://pytorch.org/)
[![torchaudio](https://img.shields.io/badge/torchaudio-2.9.0-lightgreen)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.24.0-teal)](https://pytorch.org/)
[![torchinfo](https://img.shields.io/badge/torchinfo-1.8.0-blueviolet)](https://pypi.org/project/torchinfo/1.8.0/)
[![torchmetrics](https://img.shields.io/badge/torchmetrics-1.8.2-lightgray)](https://pypi.org/project/torchmetrics/1.8.2/)

### 📦 Install Dependencies

#### 🖥️ Platform Compatibility

| **Operating System**  | **GPU Type** | **Recommended Installation** |
|:---------------------:|:------------:|:----------------------------:|
| Windows/Linux/macOS   | No GPU       | CPU version                  |
| Windows/Linux         | NVIDIA GPU   | CUDA version                 |
| Linux                 | AMD GPU      | ROCm version                 |
| macOS (Apple Silicon) | Apple GPU    | CPU version*                 |

\* macOS uses Metal Performance Shaders (MPS) backend automatically when available.

> [!TIP]
> Check the official [PyTorch installation guide](https://pytorch.org/get-started/locally/ ) to find the best option for your specific system configuration.
>
> - [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html )
> - [HIP (ROCm) semantics](https://docs.pytorch.org/docs/stable/notes/hip.html )
> - [Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html )
> - [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html )

#### 📦 Method 1: uv (**Recommended** ✅)

- Use [**uv**](https://docs.astral.sh/uv/) for dependency management. It handles dependencies, virtual environments, and locking versions more efficiently than pip.  
- To install exact dependency versions specified in [**uv.lock**](./uv.lock) for consistent environments **without** installing the current project as a package:

  ```bash
  uv sync --no-install-project --extra cpu   # Install PyTorch with CPU-only support
  uv sync --no-install-project --extra cuda  # Install PyTorch with CUDA GPU support
  # Note: For ROCm version, visit https://docs.astral.sh/uv/guides/integration/pytorch/
  ```

#### 📦 Method 2: Pip

- Install all dependencies listed in [**requirements.txt**](./requirements.txt) using [**pip**](https://pip.pypa.io/en/stable/installation/):

  ```bash
  pip install -r requirements.txt
  ```

> [!WARNING]
> The [**requirements.txt**](./requirements.txt) includes **CUDA-enabled** PyTorch by **default**. For different platforms, modify the `--extra-index-url` in [**requirements.txt**](./requirements.txt).

> [!CAUTION]
> Pip does not lock dependency versions as strictly as `uv`. You may encounter version conflicts or reproducibility issues. Only use this method if you know what you are doing.

### 🛠️ Usage Instructions

1. Open the root folder with [**VS Code**](https://code.visualstudio.com/) (`Ctrl/Cmd + K` followed by `Ctrl/Cmd + O`).
1. Open `.ipynb` files using the [**Jupyter extension**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) integrated with **VS Code**.
1. Select the correct Python kernel and virtual environment where the dependencies were installed.
1. Allow **VS Code** to install any recommended dependencies for working with Jupyter Notebooks.

> [!NOTE]
>
> - It is **highly recommended** to stick with the exact dependency versions specified in [**uv.lock**](./uv.lock) or [**requirements.txt**](./requirements.txt) rather than using the latest package versions. The repository has been **tested** on these versions to ensure **compatibility** and **stability**.
> - This repository is **actively maintained**, and dependencies are **updated regularly** to the latest **stable** versions.
> - The **table of contents** embedded in the **notebooks** may not function correctly on **GitHub**.
> - For an improved experience, open the notebooks **locally** or view them via [**nbviewer**](https://nbviewer.org/github/mr-pylin/pytorch-workshop).

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
  - Link: [docs.pytorch.org/docs/stable/index.html](https://docs.pytorch.org/docs/stable/index.html)
- **TorchVision Documentations**:
  - The torchvision package [part of the PyTorch] consists of popular datasets, model architectures, and common image transformations for computer vision.
  - Link: [docs.pytorch.org/vision/stable/index.html](https://docs.pytorch.org/vision/stable/index.html)
- **TorchAudio Documentation**:
  - The torchaudio package [part of the PyTorch] consists of audio I/O and signal processing functionalities, enabling efficient loading, transforming, and manipulating audio.
  - Link: [docs.pytorch.org/audio/stable/index.html](https://docs.pytorch.org/audio/stable/index.html)

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

## 🔍 Connect With Me

Any mistakes, suggestions, or contributions? Feel free to reach out:

- 📍 All links: [**linktr.ee/mr_pylin**](https://linktr.ee/mr_pylin)
- 💬 Telegram Group: [**@python_topics**](https://t.me/python_topics)
- 📧 Email: [**amirhosseinheydari78@gmail.com**](mailto:amirhosseinheydari78@gmail.com)

I look forward to connecting with you! 🏃‍♂️

## 📄 License

This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to **use**, **modify**, and **distribute** this code, but you **must** include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.

### ©️ Copyright Information

- **Original Images**:
  - The images located in the [**./assets/images/original/**](./assets/images/original/) folder are licensed under the **[CC BY-ND 4.0](./assets/images/original/LICENSE)**.
  - Note: This license restricts derivative works, meaning you may share these images but cannot modify them.
- **PyTorch Assets**:
  - The images located in the [**./assets/images/pytorch/**](./assets/images/pytorch/) folder are sourced from the [**PyTorch official website**](https://pytorch.org/) and are licensed under the [**BSD 3-Clause License**](https://opensource.org/licenses/BSD-3-Clause).
- **Third-Party Assets**:
  - Additional images located in [**./assets/images/third_party/**](./assets/images/third_party/) are used with permission or according to their original licenses.
  - Attributions and references to original sources are included in the code where these images are used.
