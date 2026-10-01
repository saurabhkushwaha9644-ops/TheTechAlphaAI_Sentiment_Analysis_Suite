# Task 3: AI Sentiment Analysis Suite

An enterprise-grade web application built with Python and Streamlit that implements a deep learning pipeline for real-time sentiment extraction. This suite leverages advanced Transformer architectures to process multi-lingual text inputs, fully supporting English, Hindi, and Hinglish text structures seamlessly.

## How it Works

- **User Input Processing:** The application captures raw textual inputs provided by the user through an interactive, multi-line text area interface.
- **Model Caching Framework:** To optimize execution speed and system resources, the system implements a caching decorator that keeps the underlying AI pipeline resident in memory after the initial load.
- **Deep Learning Inference:** The text is fed directly into a pre-trained RoBERTa model via the Hugging Face pipeline architecture. The model evaluates semantic word patterns to decipher underlying emotional states.
- **Dynamic Metric Scaling:** Raw prediction outputs are parsed into three distinct sentiment categories and instantly displayed alongside a calculated mathematical confidence score and interactive visual progress metrics.

## Critical Libraries & Dependencies

- **`streamlit`** -> Manages the entire graphical web framework layer, processing text areas, buttons, rendering metrics, and setting structural page states.
- **`transformers`** -> Provides the core computational abstractions needed to download, load, and perform token inference against pre-trained deep learning transformer models.
- **`torch` (PyTorch)** -> The foundational deep learning backend engine required by the transformers pipeline to run mathematical tensor matrix multiplications on the system processor.

## File Structure

```text
Project_Directory/
├── advanced_sentiment_app.py     # Main application source code
└── README.md                     # Comprehensive project documentation
```

## Setup and Installation

1. Verify that Python is properly configured on your local machine.
2. Open your system terminal or command prompt inside the project folder.
3. Install the required deep learning and interface framework packages by running the following command:

```bash
python -m pip install streamlit transformers torch
```

## Executing the Interface

To host the analytical web utility on a local development server and interact with the application through your default browser window, enter this path-safe instruction into your terminal:

```bash
python -m streamlit run advanced_sentiment_app.py
```
