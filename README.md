🧹 AutoClean – Streamlit-Based Data Cleaning & ML App

AutoClean is an interactive Streamlit web application that automates data cleaning, preprocessing, and machine learning evaluation.
Users can upload datasets, configure cleaning strategies, compare model performance, and download cleaned data — all from a browser-based interface.

🚀 Live Demo (Streamlit Cloud)

🔗 Streamlit App URL:
(Add your Streamlit Cloud link here)

The application is deployed using Streamlit Community Cloud to ensure stability and full functionality.

📁 Project Structure
Data-Cleaning-ML/
│
├── app.py                  # Streamlit UI & background styling
├── clean.py                # AutoClean engine & ML logic
├── assets/
│   └── bg.jpg              # UI background image
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation

✨ Key Features

Upload CSV or Excel datasets

Automatic & manual data cleaning modes

Missing value handling:

KNN Imputation

Linear / Logistic Regression

Mean / Median / Mode

Outlier detection using IQR & Winsorization

Duplicate removal

Categorical encoding (One-Hot / Label)

Datetime feature extraction

Automatic detection of:

Regression problems

Classification problems

Model evaluation:

RMSE (Regression)

Accuracy (Classification)

Download cleaned dataset

Fully interactive Streamlit UI

🖥️ app.py – Streamlit Frontend

app.py defines the user interface and workflow.

Responsibilities

File upload and validation

Dataset preview

Cleaning configuration via dropdowns & sliders

Triggering AutoClean evaluation

Displaying performance metrics

Downloading the cleaned dataset

UI Styling

A custom background and glassmorphism UI are applied using embedded CSS.

add_bg_design()


The background image is embedded using Base64 encoding to ensure compatibility across deployments.

🧠 clean.py – AutoClean Engine

clean.py contains the core data cleaning and ML pipeline.

Main Components
AutoClean Class

Controls the full cleaning workflow

Supports auto and manual modes

Applies transformations in a safe, non-leaky order

Cleaning Modules

MissingValues – numerical & categorical imputation

Outliers – IQR-based detection and winsorization

Duplicates – duplicate row removal

EncodeCateg – categorical encoding

Adjust – datetime extraction and precision restoration

Machine Learning Evaluation

Automatically detects problem type

Trains:

Linear Regression (Regression tasks)

Logistic Regression (Classification tasks)

Compares manual vs auto cleaning strategies

Selects the best-performing strategy

⚠️ Deployment Note

Initial deployment attempts on AWS EC2 (t3.micro) caused:

Application crashes

Reduced functionality

Memory exhaustion during KNN imputation & model training

To ensure stability and correctness, the application is deployed on Streamlit Community Cloud, which provides a more suitable environment for interactive ML workloads.

🛠️ Requirements

Python 3.9+

Streamlit

Pandas

NumPy

Scikit-learn

Install dependencies locally:

pip install -r requirements.txt


Run locally:

streamlit run app.py

📌 Known Limitations

Large datasets may increase computation time

KNN-based imputation is memory-intensive

Streamlit reruns can impact performance on very large files

👤 Author

Bindu
Sharon

⭐ Acknowledgements

Streamlit

Scikit-learn

Pandas & NumPy
