# Fusion Plasma Machine Learning Toolkit

**Radiation-Collapse Study with SVM Analysis**

---

## Overview

This project aims to study the phenomenon of radiation collapse, utilizing Support Vector Machine (SVM) for data analysis and visualization. The main processes include collecting experimental data, preprocessing, SVM analysis, evaluating results (F1 score), and generating various plots. This will help identify correlations between important parameters in discharge phenomena and detect anomalies.

---

## Directory Structure

```
fusion-plasma-ml-toolkit/
├── config/
│   └── Makefile                   # Batch execution Makefile
├── scripts/
│   └── run_full_analysis.sh       # Complete automation script
├── data/
│   ├── makedata/                  # Data collection & preprocessing
│   │   ├── plasma_data_collector.py     # Data collection & CSV output
│   │   ├── shot_data_visualizer.py      # Data visualization
│   │   ├── hysteresis_plotter.py        # Hysteresis plots
│   │   ├── classes/              # Data processing classes
│   │   └── get_params/           # Parameter acquisition
│   ├── processed/                # Processed data
│   └── raw/                      # Raw data
├── src/
│   ├── analysis/                 # Data analysis
│   │   ├── f1_score.py          # F1 score evaluation
│   │   ├── result_plotting.py   # Result plotting
│   │   └── svm_analysis.py      # SVM analysis
│   ├── preprocessing/            # Preprocessing
│   │   ├── svm_exhaustive_search.py     # SVM optimization
│   │   └── separation_region_plotter.py # Separation region plots
│   ├── detection/               # Change point detection
│   │   └── change_detection.py
│   └── utils/                   # Utilities
│       └── common.py
├── outputs/                     # Output results
│   ├── plots/                  # Plot images
│   ├── process/               # Process results
│   └── results/              # Final results
├── docs/                     # Documentation
│   ├── README_JP.md         # Japanese detailed documentation
│   └── README_EN.md         # English documentation
└── requirements.txt         # Package dependencies
```

---

## Required Packages

The following Python packages are required to run this project. It is recommended to install them all at once using `requirements.txt`.

```bash
pip install -r requirements.txt
```

Main packages:

- numpy
- matplotlib
- pandas
- scikit-learn
- docopt
- japanize_matplotlib
- scipy
- joblib

---

## Quick Start

### 🚀 Fully Automated Execution (Recommended)

```bash
# Environment setup
pip install -r requirements.txt

# Run complete analysis automatically
./scripts/run_full_analysis.sh [YYYYMMDD]
```

### 📋 Manual Execution

#### 1. Environment Setup

```bash
pip install -r requirements.txt
```

#### 2. Data Collection

```bash
cd data/makedata
python plasma_data_collector.py
```

#### 3. SVM Analysis Execution

```bash
# Individual execution
python src/preprocessing/svm_exhaustive_search.py [YYYYMMDD] [K(1~14)]

# Batch execution (recommended)
cd config
make run [YYYYMMDD]
```

#### 4. Result Visualization and Evaluation

```bash
# F1 score evaluation
python src/analysis/f1_score.py -d [YYYYMMDD]

# Result plotting
python src/analysis/result_plotting.py

# Separation region plot creation
python src/preprocessing/separation_region_plotter.py
```

---

## Main Scripts Description

### Data Collection & Preprocessing

| File                                     | Description                                  | Usage                             |
| ---------------------------------------- | -------------------------------------------- | --------------------------------- |
| `data/makedata/plasma_data_collector.py` | Experimental data collection and CSV output  | `python plasma_data_collector.py` |
| `data/makedata/shot_data_visualizer.py`  | Data visualization for specific shot numbers | `python shot_data_visualizer.py`  |
| `data/makedata/hysteresis_plotter.py`    | Hysteresis plot generation                   | `python hysteresis_plotter.py`    |

### SVM Analysis

| File                                         | Description                              | Usage                                        |
| -------------------------------------------- | ---------------------------------------- | -------------------------------------------- |
| `src/preprocessing/svm_exhaustive_search.py` | SVM optimization using Exhaustive Search | `python svm_exhaustive_search.py [date] [K]` |
| `src/analysis/svm_analysis.py`               | SVM analysis execution                   | `python svm_analysis.py [date] [seed]`       |

### Result Evaluation & Visualization

| File                                             | Description                                 | Usage                                 |
| ------------------------------------------------ | ------------------------------------------- | ------------------------------------- |
| `src/analysis/f1_score.py`                       | Evaluation using F1 score                   | `python f1_score.py [options]`        |
| `src/analysis/result_plotting.py`                | Scatter plot of results                     | `python result_plotting.py`           |
| `src/preprocessing/separation_region_plotter.py` | 1-parameter separation region plot creation | `python separation_region_plotter.py` |

### Others

| File                                | Description                                | Usage                        |
| ----------------------------------- | ------------------------------------------ | ---------------------------- |
| `src/detection/change_detection.py` | Change point detection & anomaly detection | `python change_detection.py` |

---

## Execution Steps (Detailed)

### Step 1: Data Collection

```bash
cd data/makedata
python plasma_data_collector.py
```

- Collect data for specified shot numbers from server
- Save in CSV format

### Step 2: Data Visualization (Optional)

```bash
python shot_data_visualizer.py  # Individual shot visualization
python hysteresis_plotter.py    # Hysteresis analysis
```

### Step 3: SVM Analysis

```bash
# Method 1: Individual execution
python src/preprocessing/svm_exhaustive_search.py 20240923 1

# Method 2: Batch execution (recommended)
cd config
make run 20240923
```

### Step 4: Result Evaluation

```bash
# F1 score calculation
python src/analysis/f1_score.py -d 20240923

# Result visualization
python src/analysis/result_plotting.py

# Separation region plot
python src/preprocessing/separation_region_plotter.py
```

### Step 5: Change Point Detection (Optional)

```bash
python src/detection/change_detection.py
```

---

## Notes

- Before executing each script, ensure that the necessary data files exist in the `./outputs/results/[DATE]/` directory
- Specify date format as `YYYYMMDD` (e.g., 20240923)
- Batch execution using Makefile is recommended
- Parameter K should be specified in the range of 1~14

---

## Troubleshooting

### Common Issues

1. **ImportError**: Reinstall dependencies with `pip install -r requirements.txt`
2. **File not found**: Check data file paths and existence
3. **Memory error**: Try running with smaller datasets

### Support

For issues, please contact:

- Email: maesawa-yuta436@g.ecc.u-tokyo.ac.jp

---

## Author

- **Name**: Yuta Maesawa
- **Affiliation**: Department of Systems Innovation, School of Engineering, The University of Tokyo, Yamada Laboratory
- **Email**: maesawa-yuta436@g.ecc.u-tokyo.ac.jp
- **LinkedIn**: [Yuta Maesawa](https://www.linkedin.com/in/yuta-maesawa/)

---

## License

For detailed terms of use regarding this project, please contact the author.

---

## Contributing

Contributions to this project are welcome. Please feel free to submit bug reports, feature suggestions, or pull requests.

**日本語版ドキュメントは `README.md` をご参照ください**
