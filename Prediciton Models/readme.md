# Intent Classification Workspace

This repository contains code, models, and data for intent classification using various machine learning and deep learning approaches, including BERT, BioBERT, and hybrid models. It also includes scripts and notebooks for data augmentation and validation.

## Directory Structure

- **BERT Model**

  - Fine-tuning and evaluation of BERT for intent classification.
  - Includes training notebooks, best model checkpoints, label encoders, and performance plots.
- **BioBERT Model**

  - Fine-tuning and evaluation of BioBERT for intent classification.
  - Contains training notebooks, best model checkpoints, label encoders, performance plots, and grid search results.
- **Data Augmentation and Validation**

  - Scripts and notebooks for augmenting and validating datasets.
  - Subfolders:
    - **Data Augmentation**: Methods and scripts for generating augmented data.
    - **Dataset Validation**: Tools for validating and filtering datasets.
  - Includes requirements and .gitignore.
- **Hybrid Model**

  - (Details not shown; assumed to contain hybrid model code and results.)
- **ML Models/**

  - (Details not shown; assumed to contain traditional machine learning models for intent classification.)

## Key Files

- `requirements.txt`: Lists Python dependencies for each main component.
- `*.ipynb`: Jupyter notebooks for training, fine-tuning, augmentation, and validation.
- `*.pt`: Saved PyTorch model checkpoints.
- `label_encoder.pkl`: Serialized label encoders for intent classes.
- `random_augmented_balanced_dataset.csv`: Augmented and balanced datasets for training.

## Getting Started

* Install `pytorch` with GPU CUDA (Install GPU supported CUDA Version)
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

1. **Install Dependencies**

   - Each main folder contains its own `requirements.txt`. Install dependencies as needed:
     ```sh
     pip install -r requirements.txt
     ```
2. **Run Notebooks**

   - Open the relevant `.ipynb` files in Jupyter or Google Colab to reproduce experiments or augment data.
3. **Data Augmentation**

   - Use notebooks in `Data Augmentation and Validation/Data Augmentation/` to generate more training data.
4. **Model Training**

   - Fine-tune models using the notebooks in `BERT Model/` or `BioBERT Model/`.

## Notes

- Some notebooks are designed for Google Colab and may require mounting Google Drive.
- Model checkpoints and large datasets are included for reproducibility.

---

For more details, refer to the individual notebooks and scripts in in each directory.
