"""
ChEMBL Data Retrieval - Improved Version
Focus: Binary classification (active/inactive) based on pIC50 threshold
Improvements: Validation, error handling, quality filters, no fake features
"""

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import aiohttp
import asyncio
import xml.etree.ElementTree as ET

# Activity threshold
ACTIVITY_THRESHOLD = 6.0  # pIC50 > 6.0 is active (IC50 < 1 microM)

def validate_smiles(smiles):
    """
    Validate SMILES string can be parsed by RDKit
    
    Args:
        smiles: SMILES string
        
    Returns:
        bool: True if valid, False otherwise
    """
    if smiles is None or smiles == '':
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def compute_molecular_descriptors(smiles):
    """
    Compute molecular descriptors from SMILES
    Returns None if SMILES is invalid
    
    Args:
        smiles: SMILES string
        
    Returns:
        dict: Molecular descriptors or None
    """
    if not validate_smiles(smiles):
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumRings': Chem.rdMolDescriptors.CalcNumRings(mol),
        'num_h_donors': Descriptors.NumHDonors(mol),
        'num_h_acceptors': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'lipinski_compliant': int(
            Descriptors.MolWt(mol) <= 500 and
            Descriptors.MolLogP(mol) <= 5 and
            Descriptors.NumHDonors(mol) <= 5 and
            Descriptors.NumHAcceptors(mol) <= 10
        )
    }
    
    return descriptors

def classify_compound(smiles):
    """
    Classify compound based on molecular weight
    
    Args:
        smiles: SMILES string
        
    Returns:
        str: 'Small Molecule' or 'Peptide'
    """
    if not validate_smiles(smiles):
        return 'Unknown'
    
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    
    return 'Peptide' if mw > 500 else 'Small Molecule'

print("Step 1: Fetching data from ChEMBL...")
print(f"Activity threshold: pIC50 > {ACTIVITY_THRESHOLD}")

# Initialize ChEMBL client
activity = new_client.activity

# Query activities
# Batch configuration
BATCH_SIZE = 500
TOTAL_RECORDS = 5000
NUM_BATCHES = TOTAL_RECORDS // BATCH_SIZE

print(f"Fetching {TOTAL_RECORDS} records in {NUM_BATCHES} batches of {BATCH_SIZE}")

all_activities = []

for batch_num in range(NUM_BATCHES):
    offset = batch_num * BATCH_SIZE
    print(f"\nFetching batch {batch_num + 1}/{NUM_BATCHES} (offset: {offset})...")
    
    try:
        batch = activity.filter(
            standard_type="IC50",
            target_organism="Homo sapiens",
            assay_type="B",
            offset=offset,
            limit=BATCH_SIZE
        ).only(
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "target_chembl_id",
            "assay_chembl_id"
        )
        
        batch_list = list(batch)
        all_activities.extend(batch_list)
        print(f"  Retrieved {len(batch_list)} activities")
        
        # Small delay to avoid overwhelming server
        import time
        time.sleep(2)
        
    except Exception as e:
        print(f"  Error in batch {batch_num + 1}: {e}")
        print(f"  Continuing with next batch...")
        continue

activities = all_activities

print(f"Fetched {len(activities)} activities")

# Convert to list with validation
print("\nStep 2: Processing and validating data...")
data = []
skipped_counts = {
    'missing_ic50': 0,
    'invalid_ic50': 0,
    'missing_smiles': 0,
    'invalid_smiles': 0
}

for act in activities:
    # Validate IC50
    if act['standard_value'] is None:
        skipped_counts['missing_ic50'] += 1
        continue
    
    try:
        ic50_value = float(act['standard_value'])
        pIC50 = -np.log10(ic50_value * 1e-9)
    except (ValueError, TypeError):
        skipped_counts['invalid_ic50'] += 1
        continue
    
    # Validate SMILES
    smiles = act.get('canonical_smiles')
    if smiles is None or smiles == '':
        skipped_counts['missing_smiles'] += 1
        continue
    
    if not validate_smiles(smiles):
        skipped_counts['invalid_smiles'] += 1
        continue
    
    # Add to dataset
    data.append({
        'molecule_chembl_id': act['molecule_chembl_id'],
        'target_chembl_id': act['target_chembl_id'],
        'canonical_smiles': smiles,
        'standard_value': ic50_value,
        'pIC50': pIC50,
        'assay_chembl_id': act['assay_chembl_id'],
        'is_active': int(pIC50 > ACTIVITY_THRESHOLD)
    })

# Create DataFrame
df = pd.DataFrame(data)

print(f"\nData quality report:")
print(f"  Valid entries: {len(df)}")
print(f"  Skipped - missing IC50: {skipped_counts['missing_ic50']}")
print(f"  Skipped - invalid IC50: {skipped_counts['invalid_ic50']}")
print(f"  Skipped - missing SMILES: {skipped_counts['missing_smiles']}")
print(f"  Skipped - invalid SMILES: {skipped_counts['invalid_smiles']}")

print(f"\nClass distribution:")
print(f"  Active (pIC50 > {ACTIVITY_THRESHOLD}): {df['is_active'].sum()} ({df['is_active'].mean()*100:.1f}%)")
print(f"  Inactive (pIC50 <= {ACTIVITY_THRESHOLD}): {(1-df['is_active']).sum()} ({(1-df['is_active'].mean())*100:.1f}%)")

print(f"\nComputing molecular features...")

# Compute descriptors
descriptor_list = df['canonical_smiles'].apply(compute_molecular_descriptors)
descriptor_df = pd.DataFrame(descriptor_list.tolist())

# Add compound class
df['compound_class'] = df['canonical_smiles'].apply(classify_compound)

# Merge descriptors
df = pd.concat([df, descriptor_df], axis=1)

print(f"Features added: {list(descriptor_df.columns)}")
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Save checkpoint
df.to_csv('chembl_validated_data.csv', index=False)
print(f"\nSaved to: chembl_validated_data.csv")