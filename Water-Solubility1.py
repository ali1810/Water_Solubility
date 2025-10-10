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
    return len(Chem.GetSy
