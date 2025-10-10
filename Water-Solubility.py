import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Lipinski
from rdkit.Chem import DataStructs
import pubchempy as pcp
from bs4 import BeautifulSoup
import requests
from PIL import Image
from urllib.request import urlopen
import py3Dmol
from stmol import showmol
import xgboost as xgb
import base64

st.header("-----------Aqueous Solubility Prediction-----------")

# -------------------- Utility functions --------------------

def safe_mol(smiles):
    """Return RDKit molecule or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            mol = Chem.AddHs(mol)
        return mol
    except:
        return None

def get_charges(smiles):
    if '+' in smiles:
        return 1
    elif '-' in smiles:
        return -1
    return 0

def get_many_double_bonds(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return 0
    double_bond_count = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
    return 1 if double_bond_count > 4 else 0

def get_atom_degrees(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return np.zeros(7, dtype=int)
    sum_degree_vector = np.zeros(7)
    for atom in mol.GetAtoms():
        d = atom.GetDegree()
        if d < 7:
            sum_degree_vector[d] += 1
    return sum_degree_vector.astype(int)

def get_atom_valences(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return np.zeros(7, dtype=int)
    sum_valence_vector = np.zeros(7)
    for atom in mol.GetAtoms():
        v = min(atom.GetTotalValence(), 6)
        sum_valence_vector[v] += 1
    return sum_valence_vector.astype(int)

def get_atom_hybridization(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return np.zeros(7, dtype=int)
    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    sum_hybrid_vector = np.zeros(7)
    for atom in mol.GetAtoms():
        for i, h in enumerate(hybridizations):
            if atom.GetHybridization() == h:
                sum_hybrid_vector[i] += 1
    return sum_hybrid_vector.astype(int)

def get_aromatic_atoms(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return 0
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())

def get_bond_types(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return np.zeros(5, dtype=int)
    sum_bond_type_vector = np.zeros(5)
    bond_types_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'ZERO']
    for bond in mol.GetBonds():
        t = bond.GetBondType().name
        sum_bond_type_vector += np.array([1 if t==b else 0 for b in bond_types_list])
    return sum_bond_type_vector.astype(int)

def is_conjugated(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return 0
    return sum(1 for bond in mol.GetBonds() if bond.GetIsConjugated())

def get_bonds_in_ring(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return 0
    return len(Chem.GetSymmSSSR(mol))

def get_bond_chirality(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return np.zeros(4, dtype=int)
    sum_chirality = np.zeros(4)
    for bond in mol.GetBonds():
        chirality = bond.GetStereo()
        sum_chirality += np.array([
            1 if chirality==Chem.rdchem.BondStereo.STEREONONE else 0,
            1 if chirality==Chem.rdchem.BondStereo.STEREOANY else 0,
            1 if chirality==Chem.rdchem.BondStereo.STEREOZ else 0,
            1 if chirality==Chem.rdchem.BondStereo.STEREOE else 0
        ])
    return sum_chirality.astype(int)

def get_n_atoms(smiles):
    mol = safe_mol(smiles)
    return mol.GetNumAtoms() if mol else 0

def get_n_bonds(smiles):
    mol = safe_mol(smiles)
    return mol.GetNumBonds() if mol else 0

def get_n_rings(smiles):
    mol = safe_mol(smiles)
    return len(Chem.GetSymmSSSR(mol)) if mol else 0

def calc_mol_weight(smiles):
    mol = safe_mol(smiles)
    if mol is None:
        return None
    return Descriptors.MolWt(mol)

def get_functional_groups1(smiles):
    functional_groups = {
        'Hydroxyl Group': '[OH]',
        'Carbonyl Group': 'C=O',
        'Amide Group': 'C(=O)N',
        'Carboxyl Group': 'C(=O)[OH]',
        'Alkyl': '[R]',
        'Aromatic Rings': 'c',
        'Alkene': 'C=C'
    }
    if isinstance(smiles, str):
        smiles = [smiles]
    results = []
    for s in smiles:
        mol = safe_mol(s)
        if mol is None:
            results.append({fg:0 for fg in functional_groups})
        else:
            results.append({fg: 1 if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)) else 0
                            for fg, smarts in functional_groups.items()})
    return pd.DataFrame(results)

def fingerprint1(smiles, r=2, n=128):
    if isinstance(smiles, str):
        smiles = [smiles]
    mols = [safe_mol(s) for s in smiles]
    fingerprints = []
    for m in mols:
        if m is None:
            fingerprints.append(np.zeros(n, dtype=int))
            continue
        bi = {}
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=r, nBits=n, bitInfo=bi)
        arr = np.zeros((n,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    return pd.DataFrame(fingerprints)

def generate_features_single38(smiles):
    features = [
        get_charges(smiles),
        get_many_double_bonds(smiles),
        *get_atom_degrees(smiles),
        *get_atom_valences(smiles),
        *get_atom_hybridization(smiles),
        get_aromatic_atoms(smiles),
        *get_bond_types(smiles),
        is_conjugated(smiles),
        get_bonds_in_ring(smiles),
        *get_bond_chirality(smiles),
        get_n_atoms(smiles),
        get_n_bonds(smiles),
        get_n_rings(smiles)
    ]
    columns = [
        'charge', 'many_double_bonds', 'atoms_degree_0', 'atoms_degree_1',
        'atoms_degree_2', 'atoms_degree_3', 'atoms_degree_4', 'atoms_degree_5',
        'atoms_degree_6', 'atoms_valence_0', 'atoms_valence_1', 'atoms_valence_2',
        'atoms_valence_3', 'atoms_valence_4', 'atoms_valence_5', 'atoms_valence_6',
        'atom_hybridization_S', 'atom_hybridization_SP', 'atom_hybridization_SP2',
        'atom_hybridization_SP3', 'atom_hybridization_SP3D', 'atom_hybridization_SP3D2',
        'atom_hybridization_UNSPECIFIED', 'aromatic_atoms', 'single_bonds', 'double_bonds',
        'triple_bonds', 'aromatic_bonds', 'zero_bonds', 'conjugated_bonds', 'bonds_in_ring',
        'chirality_none', 'chirality_any', 'chirality_z', 'chirality_e', 'n_atoms', 'n_bonds', 'n_rings'
    ]
    return pd.DataFrame([features], columns=columns)

def get_pubchem_solubility(smiles):
    try:
        props = pcp.get_properties(['MolecularWeight'], smiles, 'smiles')
        cid = props[0]["CID"]
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/XML"
        data = requests.get(pubchem_url)
        html = BeautifulSoup(data.content, "xml")
        sol_heading = html.find(name='TOCHeading', string='Solubility')
        if sol_heading:
            info = sol_heading.find_next_sibling('Information')
            sol = info.find('String').string if info and info.find('String') else None
            return sol
        return None
    except:
        return None

# -------------------- Streamlit UI --------------------

smiles_input = st.text_input("Enter SMILES here:", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")

if smiles_input:
    st.subheader("Molecule Visualization (2D)")
    mol = safe_mol(smiles_input)
    if mol:
        img = Chem.Draw.MolToImage(mol)
        st.image(img)

    st.subheader("Functional Groups")
    st.dataframe(get_functional_groups1(smiles_input))

    st.subheader("Morgan Fingerprint (128-bit)")
    st.dataframe(fingerprint1(smiles_input))

    st.subheader("Calculated Descriptors (38 features)")
    st.dataframe(generate_features_single38(smiles_input))

    st.subheader("PubChem Solubility")
    st.write(get_pubchem_solubility(smiles_input))

    # Example: Prepare CSV download
    df_features = generate_features_single38(smiles_input)
    csv = df_features.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="features.csv">Download Features CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
