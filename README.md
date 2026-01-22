# Ephy Repo

A repository for processing and analyzing electrophysiological data from mouse experiments. This toolkit provides automated pipelines for spike sorting, quality control, data export, and visualization dashboards.

## Features

- **Data Loading**: Load and preprocess electrophysiological recordings from various formats.
- **Spike Sorting**: Automated spike sorting using Kilosort4 for neural spike detection and clustering.
- **Quality Control**: Interactive labeling tool for assessing data quality based on motion, noise, and amplitude thresholds.
- **ALF Export**: Export processed data to ALyx File (ALF) format compatible with the International Brain Laboratory (IBL) ecosystem.
- **Visualization Dashboards**: Generate comprehensive PNG dashboards for session overviews, including heatmaps and metrics.

## Installation

### Prerequisites
- Python 3.10
- Conda (recommended for environment management)

### Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   cd Ephy_repo
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate ephys-repo
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - `ibllib`: IBL Python library for data handling
   - `kilosort`: Spike sorting algorithm
   - `spikeinterface`: Spike analysis framework
   - `PyQt5`: GUI for QC labeling
   - Scientific libraries: `numpy`, `scipy`, `pandas`, `matplotlib`

## Usage

### Run Pipeline (`run_pipeline.py`)
Processes complete sessions: loads recordings, performs spike sorting, and exports to ALF format.

Edit the script to specify:
- `DATA_ROOT`: Path to data directory
- `DB_PATH`: Path to database file (e.g., feather file with session metadata)
- `MOUSE_LIST`: List of session IDs to process
- `KS_PARAMS`: Kilosort4 parameters

Run:
```
python run_pipeline.py
```

### Run Dashboard (`run_dashboard.py`)
Generates visualization dashboards for sessions.

Configure:
- `DATA_ROOT`: Data directory
- `OUT_ROOT`: Output directory for PNG files
- `RUN_MODE`: "ALL" or specific session list

Run:
```
python run_dashboard.py
```

### Run QC Labeler (`run_qc_labeler.py`)
Interactive GUI for quality control labeling.

Set parameters:
- `DATA_ROOT`: Data directory
- `SESSION_ID`: Specific session to label
- Thresholds for motion correction (`mc_thresh`), noise (`nc_thresh`), amplitude (`amp_thresh_uv`)

Run:
```
python run_qc_labeler.py
```

## Project Structure

- `functions/`: Core modules
  - `alf.py`: ALF export utilities
  - `dashboard.py`: Dashboard generation functions
  - `load.py`: Data loading and preprocessing
  - `paths.py`: Path building and session management
  - `qc_labeler.py`: Quality control GUI and labeling
  - `qc.py`: Quality control metrics
  - `sort.py`: Spike sorting with Kilosort4
- `notebook.ipynb`: Main analysis notebook for interactive exploration
- `test.ipynb`: Testing and development notebook
- `run_*.py`: Executable scripts for different workflows
- `environment.yml`: Conda environment specification
- `requirements.txt`: Python package requirements

## Configuration

- Data paths and parameters are configured directly in the run scripts.
- Adjust Kilosort parameters in `run_pipeline.py` for spike sorting optimization.
- Dashboard styling and output settings in `run_dashboard.py`.

## Contributing

Contributions are welcome! Please submit issues for bugs or feature requests, and pull requests for code improvements.

## License

[Specify license if applicable, e.g., MIT]