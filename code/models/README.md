# 🧠 Model Architectures

> A collection of **representative deep learning architectures** implemented in
> **PyTorch**, organized by their core architectural principles and evolution.

The goal is **not** to build an exhaustive model zoo, but to implement important
and influential architectures that provide a practical overview of modern deep
learning.

## 📚 Architecture Families

| Family | Focus | Main Applications |
| :--- | :--- | :--- |
| 🧮 **MLP** | Fully connected & token-mixing networks | General-purpose, Vision |
| 🖼️ **CNN** | Convolution & spatial feature extraction | Computer Vision |
| 🔄 **RNN** | Sequential & temporal modeling | NLP, Time Series |
| ⚡ **Transformer** | Attention-based representation learning | NLP, Vision |
| 🔐 **Autoencoder** | Representation & latent-space learning | Reconstruction, Generation |

---

### 🧮 MLP Architectures

> **Multi-Layer Perceptrons** use fully connected transformations as their
> fundamental computational building blocks.

#### 🏛️ Classic and Foundational

| Architecture | Description |
| :--- | :--- |
| **Perceptron** | Fundamental single-layer neural network |
| **Multilayer Perceptron (MLP)** | Feed-forward network composed of multiple fully connected layers |

#### ⚙️ Modern and Specialized

| Architecture | Key Idea |
| :--- | :--- |
| **ResMLP** | Residual connections in an MLP-based architecture |
| **gMLP** | Gated spatial/token mixing |
| **MLP-Mixer** | Separate token mixing and channel mixing |
| **FNet** | Fourier-based token mixing instead of self-attention |

#### 👁️ Vision-Focused MLPs

| Architecture | Key Idea |
| :--- | :--- |
| **AS-MLP** | Axial shifted-window spatial mixing |
| **S²-MLP** | Spatial-shift operations for token interaction |
| **CycleMLP** | Cycle-based spatial token mixing |
| **Hire-MLP** | Hierarchical rearrangement for spatial mixing |

### 🖼️ CNN Architectures

> **Convolutional Neural Networks** exploit local spatial structure through
> convolutional operations and have historically dominated computer vision.

#### 🏛️ Classic and Foundational

| Architecture | Key Idea |
| :--- | :--- |
| [**LeNet-5**](./cnn/lenet5.ipynb) | Early CNN for handwritten digit recognition |
| [**AlexNet**](./cnn/alexnet.ipynb) | Landmark deep CNN demonstrating the effectiveness of deep learning for image classification |

#### 🧱 Deeper and Structured

| Architecture | Key Idea |
| :--- | :--- |
| [**VGGNet**](./cnn/vggnet.ipynb) | Deep stacks of small convolutional filters |
| [**GoogLeNet**](./cnn/googlenet.ipynb) | Inception modules for multi-scale feature extraction |
| [**ResNet**](./cnn/resnet.ipynb) | Residual connections for training very deep networks |

#### ⚡ Efficient and Modern

| Architecture | Key Idea |
| :--- | :--- |
| [**DenseNet**](./cnn/densenet.ipynb) | Dense feature reuse through layer-to-layer connections |
| **MobileNet** | Depthwise separable convolutions for efficient computation |
| [**Xception**](./cnn/xception.ipynb) | Extreme form of depthwise separable convolution |
| [**EfficientNet**](./cnn/efficientnet.ipynb) | Compound scaling of depth, width, and resolution |
| **ConvNeXt** | Modernized CNN design inspired by contemporary architectures |

### 🔄 RNN Architectures

> **Recurrent Neural Networks** model sequential data by maintaining a hidden
> state across time steps.

#### 🏛️ Classic and Foundational

| Architecture | Key Idea |
| :--- | :--- |
| **Vanilla RNN** | Basic recurrent sequence modeling |
| **LSTM** | Gated memory cells for long-term dependencies |
| **GRU** | Simplified gated recurrent architecture |

#### ↔️ Advanced RNNs

| Architecture | Key Idea |
| :--- | :--- |
| **Bidirectional RNN** | Processes sequences in both directions |
| **Bidirectional LSTM** | Bidirectional processing with LSTM cells |
| **Bidirectional GRU** | Bidirectional processing with GRU cells |
| **Deep RNN** | Multiple stacked recurrent layers |

#### 🔁 Sequence-to-Sequence

| Architecture | Key Idea |
| :--- | :--- |
| **Seq2Seq** | Encoder-decoder architecture for sequence transformation |
| **Encoder–Decoder LSTM** | Seq2Seq architecture based on LSTM networks |

### ⚡ Transformer Architectures

> **Transformers** use attention mechanisms to model relationships between
> elements of a sequence or spatial representation.

#### 🏛️ Classic and Foundational

| Architecture | Key Idea |
| :--- | :--- |
| **Transformer** | Original attention-based encoder-decoder architecture |
| **BERT** | Bidirectional Transformer encoder |
| **GPT** | Autoregressive Transformer for generative modeling |

#### 👁️ Vision Transformers

| Architecture | Key Idea |
| :--- | :--- |
| **ViT** | Applies Transformer encoders to image patches |
| **DeiT** | Data-efficient training of Vision Transformers |
| **Swin Transformer** | Hierarchical representations with shifted-window attention |
| **BEiT** | Masked image modeling for visual representation learning |

#### 🚀 Efficient and Specialized

| Architecture | Key Idea |
| :--- | :--- |
| **MobileViT** | Combines convolution with Transformer-based representation learning |
| **EfficientFormer** | Efficiency-oriented Transformer architecture |
| **MaxViT** | Combines local and global attention using multi-axis attention |

### 🔐 Autoencoder Architectures

> **Autoencoders** learn representations by encoding inputs into latent spaces
> and reconstructing them.

#### 🏛️ Classic and Foundational

| Architecture | Key Idea |
| :--- | :--- |
| **Autoencoder (AE)** | Encoder-decoder reconstruction |
| **Denoising Autoencoder (DAE)** | Reconstructs clean inputs from corrupted inputs |
| **Sparse Autoencoder (SAE)** | Encourages sparse latent representations |

#### 🎲 Probabilistic and Generative

| Architecture | Key Idea |
| :--- | :--- |
| **Variational Autoencoder (VAE)** | Probabilistic latent representation |
| **β-VAE** | Encourages more disentangled latent representations |
| **VQ-VAE** | Learns discrete latent representations through vector quantization |
