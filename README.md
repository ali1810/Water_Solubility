
# Water Solubility Prediction

## Introduction
This project predicts the water solubility of molecules using descriptors ,molecular fingerprints, functional groups and 38 another features (total 298 features)and machine learning models like XGBoost. 

---
## Applicability Domain

The applicability domain of this model refers to the range of chemical structures and solubility values that the model can predict accurately. Here are a few important considerations:

- **Training Data Coverage**: The model was trained on a dataset of molecules with known solubility in water. It works best for chemicals that are structurally similar to the compounds in the dataset. For example, it performs well for small organic molecules but may struggle with highly complex or unusual molecules not represented in the training data.
- **Molecular Descriptors**: The model uses molecular descriptors and fingerprints, which are effective for capturing general molecular properties. However, if a new compound has very unusual features or lacks specific molecular groups, predictions may be less reliable.
- **Outliers**: Predictions for compounds that are outliers in terms of their molecular properties (e.g., very large molecules or molecules with uncommon functional groups) should be treated with caution.
- 
## Features
- Use of molecular decriptors ,fingerprints, functional groups  and some features like chirality and others .
- XGBoost model for solubility prediction.
- Cleaned and processed dataset with molecular properties.

---

## How to Set Up and Run the Project Locally

Follow the steps below to clone the repository, set up the virtual environment, and run the Streamlit application.

### Step 1: Clone the Repository
Start by cloning the repository to your local machine. Open your terminal and run:

```bash
git clone https://github.com/ali1810/Water_Solubility.git

### Step 2: Navigate to the Project Directory

```bash
cd Water_Solubility

### Step 3: Set Up a Virtual Environment

```bash
python -m venv venv
### on windows
```bash
venv\\Scripts\\activate
### on Mac
```bash
source venv/bin/activate

### Step 4: Install Required Dependencies
```bash
pip install -r requirements.txt

### Step 5: Run the Streamlit Application
```bash
streamlit run main1.py



