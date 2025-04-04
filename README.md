


# Water Solubility Prediction

## Overview
This repository provides a machine learning-based model for predicting the water solubility of compounds using molecular descriptors and fingerprints. The model can be run locally to generate predictions for new SMILES strings.

## Features
- Accepts **SMILES strings** as input for solubility prediction.
- Uses **molecular descriptors and fingerprints** to generate predictions.
- Includes **preprocessing steps** to ensure compatibility with the model.
- Provides a ready-to-use **Streamlit application** for user-friendly execution.

### Model Evaluation

The best-performing model and its parameters were selected based on minimizing the Mean Absolute Error (MAE). We have also evaluated our model on 5 experimented data and compaare with graph how the model behave on train and test data detail has givben in the notebook 'model_evaluation_test_data.ipynb' 


## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/ali1810/Water_Solubility.git
cd Water_Solubility
```

### 2. Create a Virtual Environment
#### On macOS/Linux:  ( Make sure to run with python version 3.9.5 to match the predited result )
```bash
python -m env env
source env/bin/activate
```
#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Model

### Streamlit Cloud Version
Along with the local version, we also provide a **Streamlit Cloud** deployment of the application. You can access it online without any installation. [Streamlit Cloud Application](https://aqua-solubility-prediction.streamlit.app/)

### 1. Start the Streamlit Application
```bash
streamlit run Water-Solubility.py
```

### 2. Input New SMILES Strings for Prediction
- Enter the **SMILES strings** in the input field of the Streamlit app.
- Click the submit button to preprocess and predict solubility.
- There is also an option to upload a **CSV file** containing SMILES. Ensure that the column name for SMILES in the file is **'SMILES'**.

## Model Explanation
- The model utilizes **molecular descriptors and fingerprints** extracted from the input SMILES.
- It preprocesses the data to align with the training dataset structure.
- Predictions are generated based on learned patterns from the training data.

## Compatibility & Preprocessing
- Ensure that the input **SMILES strings** are valid.
- The preprocessing pipeline standardizes molecules before descriptor calculation.
- Missing or invalid inputs may lead to errors during prediction.

## License
This project is licensed under the **MIT License**.

## Contact
For any questions or further information, feel free to reach out:
- **Mushtaq Ali** - [mushtaq.ali@kit.edu](mailto:mushtaq.ali@kit.edu)
- **Nicole Jung** - [nicole.jung@kit.edu](mailto:nicole.jung@kit.edu)
- **Institution**: [https://www.ibcs.kit.edu](https://www.ibcs.kit.edu)

## Conclusion
This repository provides an efficient way to predict water solubility based on molecular structures. The combination of machine learning and molecular descriptors ensures reliable predictions. For any issues or improvements, feel free to open an issue in this repository.





