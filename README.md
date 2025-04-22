# Multi-Modal-Modeling-of-a-Compound-BioActivity
Multi-Modal Modeling of Compound Bioactivity: A Comparative Study of Regression and Classification Approaches


**Course:** IE7275 - Data Mining in Engineering  
**Project Team:** Dennis Mathew Jose, Venkata Siva Naga Vamsinath Thatha  
**Submission Date:** April 22, 2025  

---

## Project Overview

This project applies machine learning techniques to predict the **bioactivity** of chemical compounds, aiming to improve the **efficiency of early-stage drug discovery**. Using the **ChEMBL** database, we model both:

- **Regression** task: Predicting pIC50 values (a measure of compound potency)
- **Classification** task: Determining whether a compound is *active* or *inactive*

We explore **multi-modal data** including:
- Physicochemical properties
- Target-related descriptors
- Structural features (Morgan fingerprints)

---

## Objectives

- Predict the **pIC50** value of a compound using ML models
- Classify compounds as **active** (pIC50 ≥ 6) or **inactive**
- Interpret model predictions using SHAP values
- Reduce false negatives through **cost-benefit optimization**
- Enhance model performance via **feature engineering** and **fingerprints**

---

## Dataset

- **Source**: [ChEMBL database](https://www.ebi.ac.uk/chembl/)
- **Size**: 10,000 compound-target interactions
- **Features**:
  - Molecular descriptors: `MolWt`, `LogP`, `TPSA`, etc.
  - Biological features: `target_family`, `assay_type`, etc.
  - 512-bit Morgan fingerprints
- **Target Variables**:
  - `pIC50` (regression)
  - `activity_label` (classification using threshold pIC50 ≥ 6)

---

## Preprocessing & Feature Engineering

- Dropped columns with excessive missing values (e.g., `organism`)
- Median/mode imputation for sparse missing data
- Categorical encoding (LabelEncoder)
- Standardization of numerical features
- Variance Inflation Factor (VIF) used to address multicollinearity
- Added Morgan fingerprints for enhanced structural representation

---

## Models Used

### Regression Models
- Ridge Regression
- Random Forest Regressor
- Multi-Layer Perceptron (MLP)
- XGBoost Regressor

### Classification Models
- Logistic Regression (baseline)
- Random Forest Classifier
- MLP Classifier
- XGBoost Classifier

**Hyperparameter tuning** was performed using GridSearchCV.

---

## Evaluation Metrics

### Regression:
- R² Score
- RMSE
- MAE

### Classification:
- Accuracy, Precision, Recall, F1-score, ROC AUC
- Cost-Benefit Ratio (CBR) to prioritize minimizing false negatives

---

## Results

### Best Regression Model:
**XGBoost Regressor** (with fingerprints)  
- R² = 0.963  
- RMSE = 0.83  
- MAE = 0.60

### Best Classification Model:
**XGBoost Classifier**  
- Accuracy = 86%  
- F1-score = 0.86  
- ROC AUC = 0.94  
- CBR = 0.0833 (lowest among all models)

---

## Interpretability with SHAP

- SHAP used to explain **feature impact** globally and locally
- Without fingerprints: importance on descriptors like `MolWt`, `LogP`, `target_family`
- With fingerprints: substructures like `FP_314`, `FP_67` emerged as dominant predictors
- Waterfall plots highlight individual prediction logic

---

## Next Steps

- Scaffold-based validation to improve generalization across novel chemotypes
- Extend to multi-task learning (bioactivity, toxicity, solubility)
- Apply model to external datasets (e.g., DrugBank) for validation
- Incorporate uncertainty estimation and probabilistic thresholds for decision-making

---

## Appendix (Optional for Submission)

- GridSearch parameters used for all models
- Correlation matrix and VIF scores
- Additional SHAP waterfall plots
- Sample records after feature engineering

---

## References

- Chen, H., Engkvist, O., Wang, Y., Olivercrona, M., & Blaschke, T. (2018, January 31). The rise of Deep Learning in Drug Discovery. ScienceDirect. https://www.sciencedirect.com/science/article/pii/S1359644617303598?via%3Dihub 
- Four phases of the drug development and Discovery Process. Bioinformatics Analysis – CD Genomics. (n.d.). https://bioinfo.cd-genomics.com/resource-four-phases-of-the-drug-development-and-discovery-process.html 
- Gaulton, A., Bellis, L. J., Bento, A. P., Chambers, J., Davies, M., Hersey, A., Light, Y., McGlinchey, S., Michalovich, D., Al-Lazikani, B., & Overington, J. P. (2012). ChEMBL: a large-scale bioactivity database for drug discovery. Nucleic acids research, 40(Database issue), D1100–D1107. https://doi.org/10.1093/nar/gkr777
- Gawehn, E., Hiss, J. A., & Schneider, G. (2015, December 30). Deep learning in drug discovery - gawehn - 2016 - molecular informatics - wiley online library. Wiley Online Library. https://onlinelibrary.wiley.com/doi/10.1002/minf.201501008 
Lipinski, C. A., Lombardo, F., Domin, B. W., & Feeney, P. J. (2001, March 14). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. ScienceDirect. https://www.sciencedirect.com/science/article/abs/pii/S0169409X00001290?via%3Dihub 
- Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., Mugumbate, G., Hunter, F., Nowotka, M., Mutowo, P., Mosquera, J. F., Magariños, M. P., Félix, E., & De Veij, M. (2018, November 6). Chembl: Towards direct deposition of bioassay data. Oxford Academic. https://academic.oup.com/nar/article/47/D1/D930/5162468 
- Mervin, Lewis H., et al. “Target prediction utilising negative bioactivity data covering large chemical space.” Journal of Cheminformatics, vol. 7, no. 1, 24 Oct. 2015, https://doi.org/10.1186/s13321-015-0098-y. 
- Tse, E. G., Aithani, L., Anderson, M., Cardoso-Silva, J., Cincilla, G., Conduit, G. J., Galushka, M., Guan, D., Hallyburton, I., Irwin, B. W., Kirk, K., Lehane, A. M., Lindblom, J. C., Lui, R., Matthews, S., McCulloch, J., Motion, A., Ng, H. L., Öeren, M., … Todd, M. H. (2021). An open drug discovery competition: Experimental validation of predictive models in a series of novel antimalarials. Journal of Medicinal Chemistry, 64(22), 16450–16463. https://doi.org/10.1021/acs.jmedchem.1c00313 


---

## Environment & Tools

- Python 3.11
- Scikit-learn
- RDKit (for fingerprint generation)
- SHAP
- Pandas, NumPy, Matplotlib, Seaborn

---

## Contact

- 📧 Dennis Jose: jose.de@northeastern.edu  
- 📧 Vamsinath Thatha: thatha.v@northeastern.edu

