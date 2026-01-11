# Improving-Clinical-Usability-of-Automated-Arrhythmia-Detection-with-an-Explainable-1D-CNN


## Project Overview

This project focuses on developing an automated system for detecting cardiac arrhythmias from Electrocardiogram (ECG) signals using a deep learning approach, specifically a **1D Convolutional Neural Network (CNN)** combined with a **Long Short-Term Memory (LSTM)** network. A core objective is to enhance the **clinical usability** of the detection system by incorporating **Explainable AI (XAI)** techniques, providing insights into model predictions.

Cardiac arrhythmias, caused by electrical dysfunctions in the heart, can lead to serious health conditions. ECG is a vital diagnostic tool, but manual interpretation is time-consuming and prone to human error. This project aims to create an efficient and interpretable automated solution to assist cardiologists.

## Features

  * **ECG Signal Preprocessing:** Robust handling of raw ECG signals, including noise reduction, standardization, and beat segmentation.
  * **Hybrid Deep Learning Model:** Utilizes a custom 1D CNN for efficient feature extraction from time-series ECG data, followed by an LSTM layer to capture temporal dependencies.
  * **Multi-Class Arrhythmia Classification:** Classifies heartbeats into key arrhythmia categories (e.g., Normal, Ventricular, Supraventricular, Fusion, etc., as per AAMI standards).
  * **Explainable AI (XAI) Integration:** Implements techniques like **Grad-CAM** to visualize which parts of the ECG signal are most influential for a given arrhythmia prediction, enhancing model transparency and clinical trust.
  * **Addressing Class Imbalance:** Strategies (e.g., weighted loss) to mitigate the challenge of imbalanced datasets common in medical diagnostics, aiming to improve the detection of rare but critical arrhythmia types.
  * **Comprehensive Evaluation:** Utilizes clinically relevant metrics beyond overall accuracy, including precision, recall, and F1-score for each class.
  * **Reproducible Code:** Jupyter Notebook for step-by-step execution and a dedicated prediction script for easy demonstration.

## Dataset

This study utilizes the **MIT-BIH Arrhythmia Database**, a widely recognized benchmark dataset for arrhythmia detection.

  * The database contains 48 half-hour, two-channel ECG recordings.
  * In line with AAMI recommended practices, four recordings containing paced beats (102, 104, 107, 217) were excluded from the analysis to ensure reliable signal quality.
  * Heartbeats are categorized into classes based on AAMI standards, which are then grouped for classification:
      * **N:** Normal beat (includes N, L, R, A, a, J, S, e, j)
      * **V:** Premature ventricular contraction (V)
      * **F:** Fusion of ventricular and normal (F), fusion of paced and normal (f)
      * **E:** Ventricular escape (E)


## Model Architecture

The deep learning model is a sequential Keras model consisting of:

  * **Multiple 1D Convolutional Blocks:** Each block includes `Conv1D`, `BatchNormalization`, and `MaxPooling1D` layers to extract hierarchical features from the ECG segments.
  * **TimeDistributed Wrapper:** Applied to convolutional layers, allowing the model to process sequences of ECG segments (though currently configured for single segments, this structure allows for future expansion to multi-beat analysis).
  * **Flatten Layer:** Prepares features for the subsequent LSTM layer.
  * **LSTM Layer:** Captures temporal dependencies within the extracted features.
  * **Dense Classifier:** A final `Dense` layer with `softmax` activation for multi-class classification.


## Performance Summary

The model achieved an overall accuracy of **94%** on the test set. However, a detailed analysis of the classification report reveals:

  * **Excellent performance on the majority 'Normal' class (Class 2):** High precision (0.94) and recall (0.99).
  * **Moderate performance on Class 3 (an arrhythmia type):** Good precision (0.92) but lower recall (0.62), indicating a significant number of false negatives for this class.
  * **Critical limitation on Class 1 (a minority arrhythmia type):** The model currently shows **0.00 precision and recall** for this class, meaning it completely fails to detect these instances. This is a crucial area for future improvement to ensure clinical viability.

The project highlights the importance of evaluating models with clinically relevant metrics beyond overall accuracy, especially in the presence of severe class imbalance.


## Explainability (XAI)

To bridge the gap between model predictions and clinical understanding, a basic **Grad-CAM** (Gradient-weighted Class Activation Mapping) implementation is provided in the `predict.py` script. This technique generates a heatmap over the input ECG signal, visually indicating which time points or waveform segments were most influential in the model's decision for a particular arrhythmia classification. This allows clinicians to gain trust in the model by understanding *why* a prediction was made.


## Getting Started

Follow these steps to set up the project and reproduce the results.

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/Arrhythmia-Detection-XAI.git
cd Arrhythmia-Detection-XAI
```

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3\. Install Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not yet generated, run `pip freeze > requirements.txt` after installing all necessary packages manually, then commit it.)*
The core libraries include `tensorflow`, `keras`, `wfdb`, `numpy`, `matplotlib`, `scikit-learn`.

### 4\. Download the MIT-BIH Arrhythmia Database

The project expects the dataset to be in a `mitdb` directory within the project root.

1.  Go to the [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
2.  Download all files with extensions `.dat`, `.hea`, and `.atr` for all records.
3.  Create a folder named `mitdb` in your project's root directory:
    ```bash
    mkdir mitdb
    ```
4.  Place all the downloaded `.dat`, `.hea`, and `.atr` files into the `mitdb` folder.

### 5\. Run the Jupyter Notebook

Open the main Jupyter Notebook to see the full data preprocessing, model training, and evaluation pipeline.

```bash
jupyter notebook "Arrhythmia Detection.ipynb"
```

Execute all cells sequentially. Note that training the model may take some time depending on your hardware.

## How to Use the Prediction Script (`predict.py`)

The `predict.py` script allows you to load a trained model and make predictions on individual ECG beats, along with visualizing the Grad-CAM explanation.

### 1\. Ensure Model is Saved

After running the Jupyter Notebook, a trained model will be saved (e.g., `model-00006-0.159-0.952-0.002-0.961.weights.h5` or `.keras`). Make sure this file exists in your project directory (or in a subdirectory like `saved_models/`).

### 2\. Run the Prediction Script

You can run the `predict.py` script from your terminal. By default, it will load a specific sample ECG beat from a predefined record.

```bash
python predict.py
```

The script will output the predicted arrhythmia class and display a plot of the ECG segment with the Grad-CAM heatmap overlay.

## Future Work

  * **Address Class Imbalance More Robustly:** Implement advanced techniques like SMOTE, ADASYN, or more sophisticated sampling strategies to significantly improve recall for minority arrhythmia classes.
  * **Hyperparameter Optimization:** Conduct systematic hyperparameter tuning (e.g., using Keras Tuner, Optuna) to find optimal model configurations.
  * **Refine XAI Visualizations:** Develop more interactive and clinically intuitive visualizations for explanations.
  * **Uncertainty Quantification:** Integrate methods to quantify model uncertainty, providing clinicians with a measure of prediction reliability.
  * **Deployment as Web Application:** Build a simple web interface (e.g., using Flask/Streamlit) to allow users to upload ECG data and receive real-time predictions and explanations.
  * **Real-time Inference Optimization:** Optimize the model for faster inference times for potential real-time monitoring applications.
  * **Explore Other XAI Techniques:** Investigate LIME, SHAP, or attention-based mechanisms for further interpretability.
  * **Longer Sequence Analysis:** Adapt the model to analyze longer ECG sequences or multiple consecutive beats to capture broader rhythm patterns.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

[Ayush Sahu] - [ayushsahu1430@gmail.com / https://www.linkedin.com/in/ayush-sahu1430/]

## Acknowledgements

  * The MIT-BIH Arrhythmia Database from PhysioNet.
  * The Keras and TensorFlow communities for their powerful deep learning frameworks.
