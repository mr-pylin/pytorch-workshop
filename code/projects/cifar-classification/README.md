# 🖼️ Image Classification using CIFAR Dataset

- This project demonstrates the classification of **natural images** using the **CIFAR** dataset, which includes both **CIFAR-10** and **CIFAR-100** variants.
- The objective of the project is to apply machine learning techniques to classify images into multiple categories based on their features.

## 📥 Dataset

### CIFAR-10

- The **CIFAR-10** dataset consists of **60,000 color images** of size **32x32 pixels**, categorized into **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- For **detailed information** about the dataset, including the **structure**, **file descriptions** (e.g., training and test sets), **download link**, and more, please refer to [**CIFAR-10 Dataset README**](https://github.com/mr-pylin/datasets/tree/main/data/computer-vision/image-classification/cifar-10/README.md).

## 📝 Notebooks

### CIFAR-10

- [**Implementation 1**](./cifar-10/implementation-1/)

  <table style="margin:0 auto; border: 1px solid;">
    <tbody>
      <tr>
        <td><strong>Input Features</strong></td>
        <td colspan="6">Scaled (divided by 255), then standardized based on train set mean and std</td>
      </tr>
      <tr>
        <td><strong>Model Architecture</strong></td>
        <td colspan="6">A simple CNN (Convolutional Neural Network)</td>
      </tr>
      <tr>
        <td><strong>Dataset Split</strong></td>
        <td colspan="6">Pre-split dataset with further random division of the training set into 90% for training and 10% for validation</td>
      </tr>
      <tr>
        <td><strong>Runtime</strong></td>
        <td colspan="6">~5 minutes and 16 seconds on GTX 1650</td>
      </tr>
      <tr>
        <td rowspan="2"><strong>Train Results</strong></td>
        <td><strong>Train Metrics</strong></td>
        <td><strong>Validation Metrics</strong></td>
        <td><strong>Test Metrics</strong></td>
        <td><strong>Test Confusion Matrix</strong></td>
        <td><strong>Metrics Plot</strong></td>
        <td><strong>Demo Features</strong></td>
      </tr>
      <tr>
        <td style="text-align:center;"><a href="./cifar-10/implementation-1/results/train_val_metrics.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./cifar-10/implementation-1/results/train_val_metrics.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./cifar-10/implementation-1/results/test_metrics.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./cifar-10/implementation-1/results/test_top_1_confusion_matrix.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./cifar-10/implementation-1/results/train_val_metrics.svg">SVG</a></td>
        <td style="text-align:center;"><a href="./cifar-10/implementation-1/results/transformed_testset_demo.png">PNG</a></td>
      </tr>
    </tbody>
  </table>
