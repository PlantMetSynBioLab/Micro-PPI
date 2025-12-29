# Micro-PPI
Microenvironment-Guided Discovery and Validation of Plant Protein Interactomes

### Project Structure
```
Micro-PPI/
├── assets/          # Images, diagrams, or static files for documentation
├── configs/         # Configuration files (e.g., hyperparameters, paths)
├── data/            # Datasets (raw and processed data)
├── model/           # Model architecture definitions (neural networks)
├── results/         # Output  prediction results
├── scripts/         # Utility scripts (data preprocessing, analysis)
└── README.md        # Project documentation
```

### Installation
```
git clone https://github.com/PlantMetSynBioLab/Micro-PPI.git
cd Micro-PPI
```

#### It is recommended to create a virtual environment
```
conda create -n microppi python=3.8
conda activate microppi
```

#### Install requirements 
```
pip install -r requirements.txt
```
The default PyTorch and CUDA (cudatoolkit) versions are specified in `environment.yml`. Please adjust these version numbers according to your configuration.

### Usage
Simply replace `example_test_data.csv` in the `/data/` directory with your own test data file according to the required format.