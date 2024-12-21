# 🔢 Digit Classification using MNIST Dataset

- This project demonstrates the classification of **handwritten digits** using the **MNIST** dataset.
- The objective of the project is to apply machine learning techniques to classify digits (0–9) based on their image features.

## 📥 Dataset

- The **MNIST** dataset consists of **70,000 grayscale images** of size **28x28 pixels**, representing digits **0 to 9**.
- For **detailed information** about the dataset, including the **structure**, **file descriptions** (e.g., training and test sets), **download link**, and more, please refer to [**MNIST Dataset README**](https://github.com/mr-pylin/datasets/tree/main/data/computer-vision/image-classification/mnist/README.md).

## 📝 Notebooks

- [**Implementation 1**](./implementation-1/)

  <table style="margin:0 auto; border: 1px solid;">
    <tbody>
      <tr>
        <td><strong>Input Features</strong></td>
        <td colspan="6">Scaled (divided by 255), then standardized based on train set mean and std</td>
      </tr>
      <tr>
        <td><strong>Model Architecture</strong></td>
        <td colspan="6">A simple MLP (Multilayer Perceptron)</td>
      </tr>
      <tr>
        <td><strong>Dataset Split</strong></td>
        <td colspan="6">Pre-split dataset with further random division of the training set into 90% for training and 10% for validation</td>
      </tr>
      <tr>
        <td><strong>Runtime</strong></td>
        <td colspan="6">~3 minutes and 45 seconds on GTX 1650</td>
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
        <td style="text-align:center;"><a href="./implementation-1/results/train_val_metrics.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./implementation-1/results/train_val_metrics.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./implementation-1/results/test_metrics.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./implementation-1/results/test_top_1_confusion_matrix.csv">CSV</a></td>
        <td style="text-align:center;"><a href="./implementation-1/results/train_val_metrics.svg">SVG</a></td>
        <td style="text-align:center;"><a href="./implementation-1/results/transformed_testset_demo.png">PNG</a></td>
      </tr>
    </tbody>
  </table>
