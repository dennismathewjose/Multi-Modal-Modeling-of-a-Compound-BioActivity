"""
Data Processor - Convert extracted data to ChEMBL-compatible format
Generates molecular features and aligns schema with existing pipeline
"""

import pandas as pd
import numpy as np
import json

class DataProcessor:
    """
    Processes extracted paper data into ChEMBL-compatible format
    """
    
    def __init__(self):
        """Initialize data processor"""
        print("Data Processor initialized")
    
    def load_extracted_data(self, json_path):
        """
        Load VLM-extracted data from JSON
        
        Args:
            json_path: Path to extracted_data.json
            
        Returns:
            Dictionary of compound data
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} compounds from {json_path}")
        return data
    
    def load_smiles_mapping(self, csv_path):
        """
        Load compound ID to SMILES mapping
        
        Args:
            csv_path: Path to compound_smiles.csv
            
        Returns:
            Dictionary mapping compound_id to smiles
        """
        df = pd.read_csv(csv_path)
        mapping = dict(zip(df['compound_id'], df['smiles']))
        print(f"Loaded SMILES for {len(mapping)} compounds")
        return mapping
    
    def convert_ic50_to_pic50(self, ic50_nm):
        """
        Convert IC50 (nM) to pIC50
        
        Args:
            ic50_nm: IC50 value in nanomolar
            
        Returns:
            pIC50 value
        """
        if ic50_nm is None or ic50_nm <= 0:
            return None
        return -np.log10(ic50_nm * 1e-9)
    
    def expand_to_rows(self, extracted_data, smiles_mapping):
        """
        Expand nested data structure to rows
        Each (compound, IC50_variant) becomes a separate row
        
        Args:
            extracted_data: Dict from VLM extraction
            smiles_mapping: Dict mapping compound_id to SMILES
            
        Returns:
            List of row dictionaries
        """
        rows = []
        
        for compound_id, ic50_variants in extracted_data.items():
            # Get SMILES for this compound
            smiles = smiles_mapping.get(compound_id)
            
            if smiles is None:
                print(f"Warning: No SMILES found for compound {compound_id}, skipping")
                continue
            
            # Create row for each IC50 variant
            for variant_name, ic50_value in ic50_variants.items():
                # Skip if no data
                if ic50_value is None:
                    continue
                
                # Convert to pIC50
                pic50 = self.convert_ic50_to_pic50(ic50_value)
                
                if pic50 is None:
                    continue
                
                # Create row
                row = {
                    'molecule_chembl_id': f'PAPER_{compound_id}',
                    'canonical_smiles': smiles,
                    'standard_value': ic50_value,
                    'pIC50': pic50,
                    'target_variant': variant_name,
                    'source': 'paper'
                }
                
                rows.append(row)
        
        print(f"Expanded to {len(rows)} training examples")
        return rows
    
    def process(self, extracted_json_path, smiles_csv_path, output_path):
        """
        Main processing pipeline
        
        Args:
            extracted_json_path: Path to VLM extracted data
            smiles_csv_path: Path to SMILES mapping
            output_path: Path to save processed data
            
        Returns:
            Processed DataFrame
        """
        # Load data
        extracted_data = self.load_extracted_data(extracted_json_path)
        smiles_mapping = self.load_smiles_mapping(smiles_csv_path)
        
        # Expand to rows
        rows = self.expand_to_rows(extracted_data, smiles_mapping)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Save
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        
        return df

# Test when run directly
if __name__ == "__main__":
    print("Data processor component created")
    print("Requires SMILES mapping to run full pipeline")