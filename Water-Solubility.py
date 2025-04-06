import streamlit as st 
import json
from PIL import Image
from stmol import showmol
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import py3Dmol
import re
from sklearn.covariance import LedoitWolf
import numpy as np 
from rdkit.Chem import AllChem
import pubchempy as pcp
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
from rdkit import Chem
#from rdkit.Chem import Draw 
import xgboost
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import base64
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
import streamlit as st
import base64
from streamlit_shap import st_shap
import shap
from xgboost import XGBRegressor
import xgboost as xgb
from urllib.request import urlopen
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
import requests
import streamlit as st
from bs4 import BeautifulSoup
import deepchem as dc
from scipy.spatial.distance import mahalanobis
from scipy.stats import median_abs_deviation
import rdkit
print(rdkit.__version__)
                
#from footer import render_footer
#render_footer()
with open("feature_order.json", "r") as f:
    expected_order = json.load(f)
st.header("-----------Aqueous Solubility Prediction-----------")
import sys
import streamlit as st
import xgboost as xgb
# Output the Python version
import numpy as np
st.write("Numpy version:", np.__version__)
#st.write(f"Python Version: {sys.version}")
st.write("XGBoost version:", xgb.__version__)
st.write("hello")
st.write(rdkit.__version__)
import deepchem as dc
import streamlit as st
st.write("DeepChem version:", dc.__version__)
def calculate_aromatic_proportion(smiles):
    # Parse SMILES string and generate molecular representation
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES
    
    # Identify aromatic atoms
    aromatic_atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    
    # Calculate aromatic proportion
    total_atoms = mol.GetNumAtoms()
    aromatic_proportion = len(aromatic_atoms) / total_atoms
    
    return aromatic_proportion
####  function for handling single smiles 
def calculate_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")
    
    # List of RDKit descriptors
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptor_funcs = [desc[1] for desc in Descriptors._descList]
    
    # Compute descriptors
    features = [func(mol) for func in descriptor_funcs]
    
    # Create DataFrame
    df = pd.DataFrame([features], columns=descriptor_names)
    return df
#### Function for list of the smiles 
def calculate_rdkit_features1(smiles_list):
    """
    Calculate RDKit descriptors for a list of SMILES strings.

    Parameters:
        smiles_list (list of str): List of SMILES strings.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a SMILES string
                      and each column corresponds to a descriptor.
    """
    # List of RDKit descriptors
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptor_funcs = [desc[1] for desc in Descriptors._descList]

    data = []
    invalid_smiles = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            invalid_smiles.append((i, smiles))
            continue
        
        # Compute descriptors
        features = [func(mol) for func in descriptor_funcs]
        data.append(features)

    # Create DataFrame
    df = pd.DataFrame(data, columns=descriptor_names, index=[i for i, smiles in enumerate(smiles_list) if smiles not in dict(invalid_smiles)])

    # Log invalid SMILES strings
    if invalid_smiles:
        print(f"Invalid SMILES found at indices: {[idx for idx, sm in invalid_smiles]}\nInvalid SMILES: {[sm for idx, sm in invalid_smiles]}")

    return df
### function for single smiles and list of the smiles 
def fingerprint1(smiles, r, n): 
    # Check if input is a single SMILES string or a list
    if isinstance(smiles, str):
        smiles = [smiles]  # Convert single SMILES string to a list

    # Convert SMILES strings to RDKit molecules
    mols = [Chem.MolFromSmiles(SMILES_string) for SMILES_string in smiles]

    # Compute fingerprints
    fingerprints_array = []
    for m in mols:
        bi = {}
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=r, nBits=n, bitInfo=bi)
        array = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints_array.append(array)

    # Convert to a DataFrame with bits as columns
    fingerprints_df = pd.DataFrame(fingerprints_array)

    # Return the DataFrame for a single SMILES
    return fingerprints_df
def get_charges(smiles):
    if '+' in smiles:
        return 1
    elif '-' in smiles:
        return -1
    else:
        return 0

def get_many_double_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    double_bond_count = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    return 1 if double_bond_count > 4 else 0

def get_atom_degrees(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_degree_vector = np.zeros(7)
    for bond in bonds:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atom_degree_vector = np.array([1 if atom1.GetDegree() == d else 0 for d in range(7)])
        sum_degree_vector += atom_degree_vector
        atom_degree_vector = np.array([1 if atom2.GetDegree() == d else 0 for d in range(7)])
        sum_degree_vector += atom_degree_vector
    return sum_degree_vector.astype(int)

def get_atom_valences(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_valence_vector = np.zeros(7)
    for bond in bonds:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atom_valence_vector = np.array([1 if atom1.GetTotalValence() == v else 0 for v in range(7)])
        sum_valence_vector += atom_valence_vector
        atom_valence_vector = np.array([1 if atom2.GetTotalValence() == v else 0 for v in range(7)])
        sum_valence_vector += atom_valence_vector
    return sum_valence_vector.astype(int)

def get_atom_hybridization(smiles):
    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2, 
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D, 
        Chem.rdchem.HybridizationType.SP3D2, 
        Chem.rdchem.HybridizationType.UNSPECIFIED]
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_hybrid_vector = np.zeros(7)
    for bond in bonds:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atom_hybrid_vector = np.array([1 if atom1.GetHybridization() == h else 0 for h in hybridizations])
        sum_hybrid_vector += atom_hybrid_vector
        atom_hybrid_vector = np.array([1 if atom2.GetHybridization() == h else 0 for h in hybridizations])
        sum_hybrid_vector += atom_hybrid_vector
    return sum_hybrid_vector.astype(int)

def get_aromatic_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_aromatic_vector = np.zeros(1)
    for bond in bonds:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        atom_aromatic_vector = np.array([1 if atom1.GetIsAromatic() else 0])
        sum_aromatic_vector += atom_aromatic_vector
        atom_aromatic_vector = np.array([1 if atom2.GetIsAromatic() else 0])
        sum_aromatic_vector += atom_aromatic_vector
    return sum_aromatic_vector.astype(int)[0]

def get_bond_types(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_bond_type_vector = np.zeros(5)
    for bond in bonds:
        bond_type = bond.GetBondType().name
        bond_type_vector = np.array([1 if t == bond_type else 0 for t in ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'ZERO']])
        sum_bond_type_vector += bond_type_vector
    return sum_bond_type_vector.astype(int)

def is_conjugated(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_conjugated = np.zeros(1)
    for bond in bonds:
        is_conjugated = 1 if bond.GetIsConjugated() else 0
        conjugation_vector = np.array([is_conjugated])
        sum_conjugated += conjugation_vector
    return sum_conjugated.astype(int)[0]

def get_bonds_in_ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return len(Chem.GetSymmSSSR(mol))

def get_bond_chirality(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    bonds = mol.GetBonds()
    sum_chirality = np.zeros(4)
    for bond in bonds:
        chirality = bond.GetStereo()
        chirality_vector = np.array([1 if chirality == c else 0 for c in [Chem.rdchem.BondStereo.STEREONONE,
                                                                          Chem.rdchem.BondStereo.STEREOANY,
                                                                          Chem.rdchem.BondStereo.STEREOZ,
                                                                          Chem.rdchem.BondStereo.STEREOE]])
        sum_chirality += chirality_vector
    return sum_chirality.astype(int)

def get_n_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol.GetNumAtoms()

def get_n_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol.GetNumBonds()

def get_n_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return len(Chem.GetSymmSSSR(mol))
### Funntion for creating 38 features for single smiles 
def generate_features_single38(smiles):
    columns = [
        'charge', 'many_double_bonds', 'atoms_degree_0', 'atoms_degree_1',
        'atoms_degree_2', 'atoms_degree_3', 'atoms_degree_4', 'atoms_degree_5',
        'atoms_degree_6', 'atoms_valence_0', 'atoms_valence_1', 'atoms_valence_2',
        'atoms_valence_3', 'atoms_valence_4', 'atoms_valence_5', 'atoms_valence_6',
        'atom_hybridization_S', 'atom_hybridization_SP', 'atom_hybridization_SP2',
        'atom_hybridization_SP3', 'atom_hybridization_SP3D', 'atom_hybridization_SP3D2',
        'atom_hybridization_UNSPECIFIED', 'aromatic_atoms', 'single_bonds', 'double_bonds',
        'triple_bonds', 'aromatic_bonds', 'zero_bonds', 'conjugated_bonds', 'bonds_in_ring',
        'chirality_none', 'chirality_any', 'chirality_z', 'chirality_e', 'n_atoms',
        'n_bonds', 'n_rings'
    ]

    # Calculate features for the single SMILES string
    charge = get_charges(smiles)
    many_double_bonds = get_many_double_bonds(smiles)
    atom_degrees = get_atom_degrees(smiles).tolist()
    atom_valences = get_atom_valences(smiles).tolist()
    atom_hybridizations = get_atom_hybridization(smiles).tolist()
    aromatic_atoms = get_aromatic_atoms(smiles)
    bond_types = get_bond_types(smiles).tolist()
    conjugated_bonds = is_conjugated(smiles)
    bonds_in_ring = get_bonds_in_ring(smiles)
    bond_chirality = get_bond_chirality(smiles).tolist()
    n_atoms = get_n_atoms(smiles)
    n_bonds = get_n_bonds(smiles)
    n_rings = get_n_rings(smiles)

    # Combine all features into a single row
    feature_row = [
        charge, many_double_bonds, atom_degrees[0], atom_degrees[1],
        atom_degrees[2], atom_degrees[3], atom_degrees[4], atom_degrees[5],
        atom_degrees[6], atom_valences[0], atom_valences[1], atom_valences[2],
        atom_valences[3], atom_valences[4], atom_valences[5], atom_valences[6],
        atom_hybridizations[0], atom_hybridizations[1], atom_hybridizations[2],
        atom_hybridizations[3], atom_hybridizations[4], atom_hybridizations[5],
        atom_hybridizations[6], aromatic_atoms, bond_types[0], bond_types[1],
        bond_types[2], bond_types[3], bond_types[4], conjugated_bonds,
        bonds_in_ring, bond_chirality[0], bond_chirality[1], bond_chirality[2],
        bond_chirality[3], n_atoms, n_bonds, n_rings
    ]

    # Convert the feature row into a DataFrame
    return pd.DataFrame([feature_row], columns=columns)
#### function for creating the 38 features for list of the smiles 
def generate_features38(smiles_list):
    columns = [
        'charge', 'many_double_bonds', 'atoms_degree_0', 'atoms_degree_1',
        'atoms_degree_2', 'atoms_degree_3', 'atoms_degree_4', 'atoms_degree_5',
        'atoms_degree_6', 'atoms_valence_0', 'atoms_valence_1', 'atoms_valence_2',
        'atoms_valence_3', 'atoms_valence_4', 'atoms_valence_5', 'atoms_valence_6',
        'atom_hybridization_S', 'atom_hybridization_SP', 'atom_hybridization_SP2',
        'atom_hybridization_SP3', 'atom_hybridization_SP3D', 'atom_hybridization_SP3D2',
        'atom_hybridization_UNSPECIFIED', 'aromatic_atoms', 'single_bonds', 'double_bonds',
        'triple_bonds', 'aromatic_bonds', 'zero_bonds', 'conjugated_bonds', 'bonds_in_ring',
        'chirality_none', 'chirality_any', 'chirality_z', 'chirality_e', 'n_atoms',
        'n_bonds', 'n_rings'
    ]
    
    features = []
    for smiles in smiles_list:
        charge = get_charges(smiles)
        many_double_bonds = get_many_double_bonds(smiles)
        atom_degrees = get_atom_degrees(smiles).tolist()
        atom_valences = get_atom_valences(smiles).tolist()
        atom_hybridizations = get_atom_hybridization(smiles).tolist()
        aromatic_atoms = get_aromatic_atoms(smiles)
        bond_types = get_bond_types(smiles).tolist()
        conjugated_bonds = is_conjugated(smiles)
        bonds_in_ring = get_bonds_in_ring(smiles)
        bond_chirality = get_bond_chirality(smiles).tolist()
        n_atoms = get_n_atoms(smiles)
        n_bonds = get_n_bonds(smiles)
        n_rings = get_n_rings(smiles)

        feature_row = [
            charge, many_double_bonds, atom_degrees[0], atom_degrees[1],
            atom_degrees[2], atom_degrees[3], atom_degrees[4], atom_degrees[5],
            atom_degrees[6], atom_valences[0], atom_valences[1], atom_valences[2],
            atom_valences[3], atom_valences[4], atom_valences[5], atom_valences[6],
            atom_hybridizations[0], atom_hybridizations[1], atom_hybridizations[2],
            atom_hybridizations[3], atom_hybridizations[4], atom_hybridizations[5],
            atom_hybridizations[6], aromatic_atoms, bond_types[0], bond_types[1],
            bond_types[2], bond_types[3], bond_types[4], conjugated_bonds,
            bonds_in_ring, bond_chirality[0], bond_chirality[1], bond_chirality[2],
            bond_chirality[3], n_atoms, n_bonds, n_rings
        ]
        
        features.append(feature_row)
    
    return pd.DataFrame(features, columns=columns)

### Function for 7 groups for single smiles and list of the smiles 
def get_functional_groups1(smiles):
    from rdkit import Chem
    import pandas as pd

    functional_groups = {
        # Polar functional groups
        'Hydroxyl Group': '[OH]',
        'Carbonyl Group': 'C=O',
        'Amide Group': 'C(=O)N',
        'Carboxyl Group': 'C(=O)[OH]',
        # Non-polar functional groups
        'Alkyl': '[R]', 
        'Aromatic Rings': 'c',
        'Alkene': 'C=C'
    }
    
    # Check if input is a single SMILES string
    if isinstance(smiles, str):
        smiles = [smiles]  # Wrap it in a list for consistency
    
    results = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:  # Ensure valid SMILES
            fg_presence = {fg: 1 if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)) else 0 for fg, smarts in functional_groups.items()}
        else:
            fg_presence = {fg: 0 for fg in functional_groups}  # Default to 0 if SMILES is invalid
        #fg_presence['SMILES'] = s
        results.append(fg_presence)
    
    # Convert results to a DataFrame
    data = pd.DataFrame(results)
    
    return data
### for single smiles 
def calc_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Descriptors.MolWt(mol)
    return None
#### for smiles list 
def calc_mol_weight1(smiles_list):
    """
    Calculate molecular weights for a list of SMILES strings.

    Parameters:
        smiles_list (list of str): List of SMILES strings.

    Returns:
        list: Molecular weights for valid SMILES strings, or None for invalid ones.
    """
    return [
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else None
        for smiles in smiles_list
    ]



###
#def app():
#st.write('**Type SMILES below**...then press predict button')
SMILES = st.text_input('**Type SMILES below**👇...then press predict button',value = "CC(=O)OC1=CC=CC=C1C(=O)O")
### Generate 125 feature engineered based on the structure of the smiles ....
mol = Chem.MolFromSmiles(SMILES)
if mol is None:
    st.markdown('<p style="color:red; font-weight:bold;">❌ Invalid SMILES received!</p>', unsafe_allow_html=True)
else: 
    st.markdown('<p style="color:green; font-weight:bold;">✅ Valid SMILES received!</p>', unsafe_allow_html=True)

def clean_data(data):
    """Fixes NaN, Infinity, and extreme values in the dataset."""
    data = np.array(data, dtype=np.float64)

    # Replace NaN with the mean (or 0 if all values are NaN)
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        mean_value = np.nanmean(data) if not np.all(nan_mask) else 0.0
        data[nan_mask] = mean_value  # Replace NaNs

    # Replace Infinity with a large finite number
    inf_mask = np.isinf(data)
    data[inf_mask] = np.sign(data[inf_mask]) * 1e6  # Replace +inf with 1e6, -inf with -1e6

    # Clip excessively large values
    data = np.clip(data, -1e6, 1e6)

    return data
if st.button("Checke Application Domain"):  # Runs only when clicked
    if not SMILES:  
        st.warning("Please enter a valid SMILES string.")
    else:
        mol_weight=calc_mol_weight(SMILES)
        # Compute descriptors
        #df125_new = calculate_rdkit_features(SMILES)
        #df125 = df125_new.iloc[:, :125]
        #df38 = generate_features_single38(SMILES)
        #df128 = fingerprint1(SMILES, 2, 128)
        #df7 = get_functional_groups1(SMILES)

        # Combine all computed descriptors          
        #new_molecule_vector = pd.concat([df125, df128, df7, df38], axis=1)    
        
        # Load saved training set vectors 
        #training_set_vectors = np.load("training_set_vectors.npy")  # Ensure it's a NumPy array
        # Compute t-SNE Embeddings (Only for Visualization)
        #new_molecule_vector=clean_data(new_molecule_vector)


### Standardize Data (Important for PCA & Mahalanobis)

        #scaler = StandardScaler()
        #training_set_vectors_scaled = scaler.fit_transform(training_set_vectors)
        #new_molecule_vector_scaled = scaler.transform(new_molecule_vector)


# Compute t-SNE Embeddings (Only for Visualization)

        #tsne = TSNE(n_components=3, random_state=42, perplexity=50)
        #training_set_vectors_tsne3 = tsne.fit_transform(training_set_vectors_scaled)

        #tsne = TSNE(n_components=3, random_state=42, perplexity=50)
        #training_set_vectors_tsne3 = tsne.fit_transform(training_set_vectors_scaled)


# Reduce Dimensionality with PCA for Mahalanobis Distance

        #pca = PCA(n_components=3)  # Use 3D PCA for meaningful distance calculations
        #training_set_vectors_pca3 = pca.fit_transform(training_set_vectors_scaled)


# Approximate New Molecule PCA Coordinates Using k-NN Regression

        #knn = KNeighborsRegressor(n_neighbors=5, weights="uniform")
        #knn.fit(training_set_vectors_scaled, training_set_vectors_pca3)
        #new_molecule_pca3 = knn.predict(new_molecule_vector_scaled).flatten()


    # Compute Mahalanobis Distance in PCA Space

# Compute covariance matrix using Ledoit-Wolf for stability
        #cov_matrix = LedoitWolf().fit(training_set_vectors_pca3).covariance_
        #inv_cov_matrix = np.linalg.inv(cov_matrix)

# Compute mean in PCA space
        #mean_pca = np.mean(training_set_vectors_pca3, axis=0)

# Compute Mahalanobis distances of training molecules
        #distances = [mahalanobis(t, mean_pca, inv_cov_matrix) for t in training_set_vectors_pca3]

# Compute Mahalanobis distance of new molecule
        #new_molecule_distance = mahalanobis(new_molecule_pca3, mean_pca, inv_cov_matrix)


#  Compute Applicability Domain Threshold

# Compute 90th percentile instead of MAD for better robustness
        #threshold_distance = np.percentile(distances, 95)

        
               # Check Applicability Domain (AD)
        #if new_molecule_distance < threshold_distance:
        #    st.markdown('<p style="color:green; font-weight:bold;">✅ The compound is within the Applicability Domain!</p>', unsafe_allow_html=True)
        #else:
        #    st.markdown('<p style="color:red; font-weight:bold;">❌ Warning: Molecule is outside the model’s applicability domain. Prediction may be unreliable!</p>', unsafe_allow_html=True)
        if mol_weight < 600:
            st.markdown('<p style="color:green; font-weight:bold;">✅ The compound is within the Applicability Domain!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:red; font-weight:bold;">❌ Warning: Molecule is outside the model’s applicability domain. Prediction may be unreliable!</p>', unsafe_allow_html=True)





        
prop=pcp.get_properties([ 'MolecularWeight'], SMILES, 'smiles')
x = list(map(lambda x: x["CID"], prop)) 
y=x[0]
    #print(y) 
x = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/%s/PNG?image_size=300x300"
url=(x % y)

img = Image.open(urlopen(url))
mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
mblock = Chem.MolToMolBlock(mol)
xyzview = py3Dmol.view(width=300,height=300)
    #xyzview = py3Dmol.view(query=′pdb:1A2C′)
xyzview.addModel(mblock,'mol')
xyzview.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
style = 'stick'
spin = st.checkbox('Animation', value = True) 
#style = st.selectbox('Chemical structure',['stick','ball-and-stick','line','cross','sphere']) 

#bcolor = st.sidebar.color_picker('Pick background Color', '#0C0C0B')
xyzview.spin(True)
if spin:
    xyzview.spin(True)
else:
    xyzview.spin(False)
    #xyzview.setStyle({'sphere':{}})
xyzview.setBackgroundColor('#EAE5E5')
xyzview.zoomTo()
xyzview.setStyle({style:{'color':'spectrum'}})
    


col1, mid, col2 = st.columns([15,2.5,15])
#col1, mid, col2 = st.columns([15,0.5,15])
with col1:
        st.image(img, use_column_width=False)
with col2:
        showmol(xyzview,height=300,width=400) 

if st.button("Predict"):
    try:
        # Retrieve PubChem data
        prop = pcp.get_properties(['MolecularWeight'], SMILES, 'smiles')
        x = list(map(lambda x: x["CID"], prop))
        y = x[0]
        pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/%s/xml"
        data = requests.get(pubchem_url % y)

        # Parse solubility data from PubChem
        html = BeautifulSoup(data.content, "xml")
        solubility = html.find(name='TOCHeading', string='Solubility')
        if solubility is None:
            sol = None
        else:
            solub = solubility.find_next_sibling('Information').find(name='String').string
            sol = solub

        # Compute molecular descriptors
        df125_new = calculate_rdkit_features(SMILES)
        df125 = df125_new.iloc[:, :125]
        df38 = generate_features_single38(SMILES)
        df128 = fingerprint1(SMILES, 2, 128)
        df7 = get_functional_groups1(SMILES)

        # Combine all computed descriptors
        combined_df = pd.concat([df125, df128, df7, df38], axis=1)
        new_features_ordered = combined_df[expected_order]
        st.write(new_features_ordered.columns)
        # Load and make predictions using the model
        loaded_model = xgb.XGBRegressor()
        loaded_model.load_model('xgboost_model_298_4045.json')
        pred_xgb = loaded_model.predict(new_features_ordered)
        #st.write(pred_xgb)
        pred_rf2 = np.round(pred_xgb, 2)
        mol_liter1 = 10**pred_xgb
        mol_liter2 = np.round(mol_liter1, 2)
        MolWt1 = calc_mol_weight(SMILES)
        Gram_liter1 = (10**pred_xgb) * MolWt1

        # Create a DataFrame for the results
        data = dict(
            SMILES=SMILES,
            Predicted_LogS=pred_rf2,
            Mol_Liter=mol_liter2,
            Gram_Liter=Gram_liter1,
            Experiment_Solubility_PubChem=sol,  # Includes None if solubility is unavailable
        )
        df = pd.DataFrame(data, index=[0])

        # Display results
        st.write('Predicted LogS values for single SMILES')
        st.table(df.style.format({
            "Predicted_LogS": "{:.2f}",
            "Mol_Liter": "{:.4f}",
            "Gram_Liter": "{:.4f}"
        }))
        st.write('Computed molecular descriptors')
        st.table(combined_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
 
def getAromaticProportion(m):
         aromatic_list = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
         aromatic = 0
         for i in aromatic_list:
           if i:
             aromatic += 1
           heavy_atom = Lipinski.HeavyAtomCount(m)
         return aromatic / heavy_atom if heavy_atom != 0 else 0
st.write("""---------**OR**---------""")
st.write("""**Upload a 'csv' file with a column named 'SMILES'** (Max:2000)""")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    # data
        SMILES=data["SMILES"]  
      
if st.button('Prediction for input file'):

           
                
            df125_new=calculate_rdkit_features1(SMILES)
                
            df125 = df125_new.iloc[:, :125]
                 
            df38=generate_features38(SMILES)
                
            df128=fingerprint1(SMILES,2,128)
                
            df7=get_functional_groups1(SMILES)
                
            combined_df = pd.concat([df125, df128, df7, df38], axis=1)
                
                #xgboost_model.save_model('data/xgboost_model1.json')
            loaded_model = xgb.XGBRegressor()  # or XGBRegressor for regression tasks
            loaded_model.load_model('xgboost_model_298_4045.json')
            pred_xgb = loaded_model.predict(combined_df)
            pred_xgb1 =  np.round(pred_xgb,2)
            #st.write(pred_xgb)	
            mol_liter1   =  10**pred_xgb
            mol_liter2   = np.round(mol_liter1,2)
            MolWt = calc_mol_weight1(SMILES)
            Gram_liter=(10**pred_xgb)*MolWt    
            df_results = pd.DataFrame(SMILES, columns=['SMILES'])
            df_results["Predicted - LogS"]=pred_xgb1
            df_results["Mol/Liter"]=mol_liter2
            df_results["Gram/Liter"]=Gram_liter
            st.write('Predicted solubility in LogS, Mol/liter and Gram/liter ')
            st.write(df_results)
            st.write('Predicted solubility ready in your csv file ')
            csv = df_results.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings
            linko= f'<a href="data:file/csv;base64,{b64}" download="aquosol_predictions.csv">Download csv file</a>'
            st.markdown(linko, unsafe_allow_html=True)

     
