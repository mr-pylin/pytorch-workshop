# 🎧 Music Genre Classification using GTZAN Dataset

- This project demonstrates the classification of music genres using the **GTZAN** dataset.
- The objective of the project is to apply machine learning techniques to classify music tracks based on their features.

## 📥 Dataset

- The **GTZAN** dataset has **1,000** audio tracks across **10 genres** (Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock).
- For **detailed information** about the dataset, including the **structure**, **file descriptions** (e.g., audio files, Mel spectrogram images), **download link**, and etc, please refer to [**GTZAN Dataset README**](https://github.com/mr-pylin/datasets/blob/main/data/audio-processing/audio-classification/gtzan/README.md).
- ⚠️ **Known Issues**: The file `jazz0054.wav`, originally corrupted, has been replaced with a corrected version mentioned in the above **README** link.

## 📝 Notebooks

- [**Implementation 1**](./implementation-1/)

  <table style="margin:0 auto; border: 1px solid;">
    <tbody>
      <tr>
        <td><strong>Input Features</strong></td>
        <td colspan="6">Mel spectrograms extracted from audio files</td>
      </tr>
      <tr>
        <td><strong>Model Architecture</strong></td>
        <td colspan="6">A simple MLP (Multilayer Perceptron)</td>
      </tr>
      <tr>
        <td><strong>Dataset Split</strong></td>
        <td colspan="6">85% train, 5% validation, 10% test</td>
      </tr>
      <tr>
        <td><strong>Runtime</strong></td>
        <td colspan="6">~34 seconds on GTX 1650</td>
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
        <td style="text-align:center;"><a href="./implementation-1/results/features_demo.png">PNG</a></td>
      </tr>
    </tbody>
  </table>
