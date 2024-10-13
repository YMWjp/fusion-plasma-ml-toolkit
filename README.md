### 仮置き

1.  →
    cd makedata
    python pr7_25.py

2.  →
    python pr8.py [date] [K(1~14)]

    or

    make run [data] で一気に実行

3.  →
    python F1score.py -d [date]

4.  →
    python pr9_5.py
    → 少し修正したもの。
    python pr9_5original.py
    → オリジナルのもの。
    python svm_result_analysis_and_plot.py
    → 簡略化等で修正したもの。

get_Isat を修正する

# Project Name

**Radiation-Collapse Study with SVM Analysis**

---

## Overview

This project aims to study the phenomenon of radiation collapse, utilizing Support Vector Machine (SVM) for data analysis and visualization. The main processes include collecting experimental data, preprocessing, SVM analysis, evaluating results (F1 score), and generating various plots. This will help identify correlations between important parameters in discharge phenomena and detect anomalies.

---

## Directory Structure

```
project_root/
├── makedata/
│   ├── pr7_25.py
│   ├── makeimage.py
│   ├── makehysteresis.py
│   ├── classes/
│   │   └── CalcMPEXP.py
│   └── ...
├── classes/
│   └── ...
├── scripts/
│   ├── svm_result_analysis_and_plot.py
│   ├── pr8.py
│   ├── pr9_5.py
│   ├── pr9_5original.py
│   ├── change_detection.py
│   ├── F1score.py
│   └── ES_SVM.py
├── common/
│   └── names_dict.py
├── results/
│   └── ...
├── hist&sccaterpng/
│   └── ...
├── README.md
└── requirements.txt
```

---

## Required Packages

The following Python packages are required to run this project. It is recommended to install them all at once using `requirements.txt`.

```bash
pip install -r requirements.txt
```

Here is a list of the main packages:

- numpy
- matplotlib
- pandas
- scikit-learn
- docopt
- japanize_matplotlib
- scipy
- joblib

---

## Data Collection and Preprocessing

### `makedata/pr7_25.py`

This script performs data collection and preprocessing. It retrieves necessary physical parameters from the server for specific shot numbers and saves them as CSV files.

**Usage:**

```bash
cd makedata
python pr7_25.py
```

### `makedata/makeimage.py`

This script visualizes each parameter for specific shot numbers (e.g., 115083) using the collected data. It generates multiple graphs and saves the results as image files.

**Usage:**

```bash
cd makedata
python makeimage.py
```

### `makedata/makehysteresis.py`

This script generates hysteresis plots by detecting outliers and smoothing time-series data for selected shot numbers.

**Usage:**

```bash
cd makedata
python makehysteresis.py
```

---

## Data Analysis

### `svm_result_analysis_and_plot.py`

This is the main script for analyzing SVM results and plotting scatter diagrams. It handles data loading, weight and bias calculations, function evaluations, and plot generation.

**Usage:**

```bash
python svm_result_analysis_and_plot.py
```

### `pr8.py`

This script uses Exhaustive Search to explore optimal SVM parameters. It performs analysis based on specified datasets and parameter lists.

**Usage:**

```bash
python pr8.py [date] [K(1~14)]
```

or

```bash
make run [data]
```

Batch execution is also possible.

### `ES_SVM.py`

This module includes the class `ExhaustiveSearch`, which implements Exhaustive Search. It explores SVM model parameters exhaustively and selects the optimal model through cross-validation.

**Usage:**

```bash
python ES_SVM.py [date] [seed]
```

### `F1score.py`

This script evaluates SVM analysis results using the F1 score and assists in selecting the optimal model. It compares F1 scores for different parameter combinations.

**Usage:**

```bash
python F1score.py [options]
```

Options include drawing DoS diagrams, processing multiple seeds, and settings for specific projects. Please refer to the documentation within the script for details.

### `change_detection.py`

This script detects change points in the data for selected shot numbers and identifies and visualizes anomalous data points.

**Usage:**

```bash
python change_detection.py
```

---

## Data Management Class

### `makedata/classes/CalcMPEXP.py`

This module includes the class `CalcMPEXP`, which performs calculations and plotting of data. It supports the calculation and visualization of various physical parameters based on the retrieved data.

---

## Other Important Files

### `common/names_dict.py`

This file defines a dictionary that stores parameter names and their descriptions used throughout the project. It is imported and used by each script.

### `README.md`

This document provides an overview of the project, usage instructions, and directory structure.

### `requirements.txt`

This file lists the Python packages required for the project.

---

## Execution Steps

1. **Environment Setup**

   Install the required packages.

   ```bash
   pip install -r requirements.txt
   ```

2. **Data Collection**

   Collect data based on shot numbers and save it as CSV files.

   ```bash
   cd makedata
   python pr7_25.py
   ```

3. **Data Visualization**

   Generate images using the collected data.

   ```bash
   python makeimage.py
   python makehysteresis.py
   ```

4. **SVM Analysis**

   Execute Exhaustive Search to explore optimal parameters for the SVM model.

   ```bash
   python pr8.py [date] [K(1~14)]
   ```

   or

   ```bash
   make run [data]
   ```

5. **Result Analysis and Plotting**

   Analyze SVM results and generate scatter plots.

   ```bash
   python svm_result_analysis_and_plot.py
   ```

6. **Evaluation**

   Evaluate analysis results using the F1 score.

   ```bash
   python F1score.py [options]
   ```

7. **Change Point Detection**

   Execute anomaly detection in the data.

   ```bash
   python change_detection.py
   ```

---

## Notes

- Before executing each script, ensure that the necessary data files are present in the `./results/[DATE]/` directory.

- For scripts like `F1score.py` and `pr8.py`, please specify the required arguments correctly at runtime. Refer to the documentation or comments within each script for detailed usage instructions.

- Adjust the parameter settings within the scripts as needed to improve data preprocessing and analysis results.

---

## Author Profile

- **Name**: Yuta Maesawa
- **Affiliation**: Department of Systems Innovation, School of Engineering, The University of Tokyo, Yamada Laboratory
- **Email**: maesawa-yuta436@g.ecc.u-tokyo.ac.jp
- **LinkedIn**: [Yuta Maesawa's LinkedIn](https://www.linkedin.com/in/yuta-maesawa/)

---

## References

- Under construction

---

## Update History (Currently in production)

- **20XX-00-00**: Initial version created

---

## Contribution Guidelines

Contributions to this project are welcome. Please feel free to submit bug reports, feature suggestions, or pull requests.

---

# Conclusion

We hope this project will help in understanding and analyzing plasma contact and non-contact states. We look forward to your cooperation and feedback.
