# Water Pump Functionality Prediction

A machine learning project for predicting the operational condition of water pumps and other waterpoints in Tanzania. The project combines exploratory data analysis, cleaning and feature engineering, supervised multiclass classification, model serialization, and a lightweight FastAPI prediction service.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Prediction Target](#prediction-target)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Feature Engineering and Cleaned Data](#feature-engineering-and-cleaned-data)
- [Modelling Approach](#modelling-approach)
- [Saved Model Artifacts](#saved-model-artifacts)
- [API Application](#api-application)
- [Getting Started](#getting-started)
- [Running the API](#running-the-api)
- [Example Prediction Request](#example-prediction-request)
- [Notebook Workflow](#notebook-workflow)
- [Recommended Evaluation Metrics](#recommended-evaluation-metrics)
- [Known Notes and Limitations](#known-notes-and-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Reliable access to functional waterpoints is essential for communities. This repository uses waterpoint metadata to predict whether a pump is currently functional, functional but in need of repair, or non-functional. These predictions can support maintenance prioritisation by governments, NGOs, and local operators.

The project includes:

- Raw feature and label datasets.
- A processed modelling dataset.
- Exploratory data analysis and correlation notebooks.
- Model training and selection notebooks.
- Serialized model artifacts.
- A FastAPI app for serving predictions.

## Problem Statement

Given information about a waterpoint's location, construction, management, payment structure, water source, water quality, quantity, and extraction mechanism, predict the waterpoint's operational status.

This is a **supervised multiclass classification** problem because each training example has a known target label and there are three possible classes.

## Prediction Target

The target variable is `status_group`.

| Class | Meaning |
| --- | --- |
| `functional` | The waterpoint is operational. |
| `functional needs repair` | The waterpoint works but requires repair. |
| `non functional` | The waterpoint is not operational. |

## Repository Structure

```text
water-pump-prediction/
├── app/
│   └── app.py                         # FastAPI prediction service
├── data/
│   ├── raw/
│   │   ├── features.csv               # Raw feature data, 59,400 rows x 40 columns
│   │   └── labels.csv                 # Raw labels, 59,400 rows x 2 columns
│   └── processed/
│       └── cleaned_data.csv           # Cleaned modelling data, 36,764 rows x 16 columns
├── models/
│   ├── best_model.pkl                 # Placeholder/empty artifact in current repo state
│   ├── best_xgb_model.joblib          # Serialized XGBoost deployment pipeline
│   └── label_encoder.joblib           # Serialized target label encoder
├── notebooks/
│   ├── EDA.ipynb                      # Raw-data EDA, merging, cleaning, feature engineering
│   ├── corr.ipynb                     # Post-cleaning EDA and statistical testing
│   ├── ml.ipynb                       # Model training, evaluation, tuning, and saving
│   └── water_pump_analysis.ipynb      # Empty notebook placeholder in current repo state
├── INSTRUCTIONS.md                    # Original internship/project brief
├── README.md                          # Project documentation
└── requirements.txt                   # Dependency file; currently empty
```

## Dataset

### Raw Files

The raw data is split into two CSV files:

- `data/raw/features.csv` contains waterpoint input features.
- `data/raw/labels.csv` contains the target label, `status_group`.

Both files share an `id` column and should be merged on `id` before analysis or model training.

### Raw Feature Examples

The raw feature file includes 40 columns covering:

- Location: `longitude`, `latitude`, `region`, `district_code`, `lga`, `ward`, `basin`.
- Construction and timing: `date_recorded`, `construction_year`, `gps_height`.
- Waterpoint setup: `extraction_type`, `extraction_type_group`, `extraction_type_class`, `waterpoint_type`, `waterpoint_type_group`.
- Water characteristics: `water_quality`, `quality_group`, `quantity`, `quantity_group`, `source`, `source_type`, `source_class`.
- Management and finance: `funder`, `installer`, `management`, `management_group`, `scheme_management`, `payment`, `payment_type`, `permit`.
- Community context: `population`, `public_meeting`.

## Feature Engineering and Cleaned Data

The processed file, `data/processed/cleaned_data.csv`, is the compact dataset used for modelling. It contains 36,764 rows and 16 columns:

| Column | Description |
| --- | --- |
| `amount_tsh` | Total static head / water availability indicator. |
| `gps_height` | Altitude of the waterpoint. |
| `population` | Population around the waterpoint. |
| `age` | Engineered feature representing waterpoint age at recording time. |
| `month_recorded` | Month extracted from `date_recorded`. |
| `permit` | Whether the waterpoint has a permit. |
| `waterpoint_type_group` | Grouped waterpoint type. |
| `source_class` | Broad water source class. |
| `quantity` | Available water quantity category. |
| `quality_group` | Grouped water quality category. |
| `payment_type` | Payment arrangement category. |
| `management_group` | Grouped management category. |
| `extraction_type_class` | Grouped extraction mechanism class. |
| `region` | Administrative region. |
| `basin` | Water basin. |
| `status_group` | Target class. |

The EDA notebook shows that cleaning included creating `age`, extracting `month_recorded`, dropping selected missing values, standardising basin values, and replacing spaces in categorical values with underscores.

## Modelling Approach

The modelling workflow in `notebooks/ml.ipynb` follows these broad steps:

1. Load `data/processed/cleaned_data.csv`.
2. Split features from the target variable.
3. Create a stratified train/test split.
4. Encode the target labels with `LabelEncoder`.
5. Build preprocessing pipelines for numeric and categorical columns.
6. Evaluate multiple candidate models, including logistic regression, decision tree, random forest, gradient boosting, XGBoost, and LightGBM-style workflows.
7. Use class-imbalance-aware evaluation such as weighted F1-score and classification reports.
8. Save the deployment model and label encoder with `joblib`.

The notebook notes that a Random Forest model performed well but produced a very large serialized artifact, so an XGBoost pipeline was saved for deployment because it was more practical to serve.

## Saved Model Artifacts

The `models/` directory contains:

| File | Purpose |
| --- | --- |
| `best_xgb_model.joblib` | Main serialized model pipeline used by the API. |
| `label_encoder.joblib` | Converts numeric model predictions back to original class labels. |
| `best_model.pkl` | Present in the repository, but currently empty. Do not rely on it without recreating it. |

## API Application

The prediction API lives in `app/app.py` and uses FastAPI. On startup it loads:

- `../models/best_xgb_model.joblib`
- `../models/label_encoder.joblib`

The app exposes:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Health/welcome endpoint. |
| `POST` | `/predict` | Accepts waterpoint features and returns `predicted_status`. |

### API Input Schema

The `/predict` endpoint expects the following JSON fields:

```json
{
  "amount_tsh": 6000.0,
  "gps_height": 1390,
  "population": 109,
  "age": 12.0,
  "month_recorded": 3,
  "permit": false,
  "waterpoint_type_group": "communal_standpipe",
  "source_class": "groundwater",
  "quantity": "enough",
  "quality_group": "good",
  "payment_type": "annually",
  "management_group": "user-group",
  "extraction_type_class": "gravity",
  "region": "Iringa",
  "basin": "Lake_Nyasa"
}
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd water-pump-prediction
```

### 2. Create and Activate a Virtual Environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

The current `requirements.txt` file is empty, so install the packages used by the notebooks and API manually or update `requirements.txt` before installation.

Suggested dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna joblib fastapi uvicorn pydantic matplotlib seaborn scipy jupyter notebook ipykernel
```

If you add these dependencies to `requirements.txt`, you can then run:

```bash
pip install -r requirements.txt
```

## Running the API

From the repository root, run:

```bash
cd app
uvicorn app:app --reload
```

Then open:

- API root: <http://127.0.0.1:8000/>
- Interactive Swagger docs: <http://127.0.0.1:8000/docs>

> Note: The model paths in `app/app.py` are relative to the `app/` directory, so starting the server from inside `app/` is the safest option unless you change the model loading paths.

## Example Prediction Request

With the API running, send a request using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "amount_tsh": 6000.0,
    "gps_height": 1390,
    "population": 109,
    "age": 12.0,
    "month_recorded": 3,
    "permit": false,
    "waterpoint_type_group": "communal_standpipe",
    "source_class": "groundwater",
    "quantity": "enough",
    "quality_group": "good",
    "payment_type": "annually",
    "management_group": "user-group",
    "extraction_type_class": "gravity",
    "region": "Iringa",
    "basin": "Lake_Nyasa"
  }'
```

Example response format:

```json
{
  "predicted_status": "functional"
}
```

The exact class returned depends on the loaded model pipeline.

## Notebook Workflow

### `notebooks/EDA.ipynb`

Use this notebook for:

- Loading raw feature and label files.
- Inspecting shapes, columns, missing values, and target distribution.
- Merging features with labels.
- Exploring numeric and categorical variables.
- Creating engineered features such as `age` and `month_recorded`.
- Cleaning and standardising categorical values.
- Saving the processed dataset.

### `notebooks/corr.ipynb`

Use this notebook for post-cleaning analysis, including:

- Univariate visualisations.
- Bivariate feature-vs-target analysis.
- Correlation analysis for numeric variables.
- Chi-square testing for categorical variables against the target.
- Kruskal-Wallis testing for numeric variables across target classes.

### `notebooks/ml.ipynb`

Use this notebook for:

- Loading cleaned data.
- Splitting training and test sets.
- Encoding labels.
- Building preprocessing pipelines.
- Running baseline model comparisons.
- Handling class imbalance.
- Hyperparameter tuning with Optuna and search strategies.
- Saving the deployment model and label encoder.

## Recommended Evaluation Metrics

Because the target classes may be imbalanced, accuracy alone can be misleading. Prefer reporting:

- Accuracy.
- Weighted F1-score.
- Macro F1-score.
- Per-class precision, recall, and F1-score.
- Confusion matrix.

These metrics help reveal whether the model performs well across all classes, including the less frequent `functional needs repair` class.

## Known Notes and Limitations

- `requirements.txt` is currently empty even though the project requires several Python packages.
- `models/best_model.pkl` is currently an empty file and should not be treated as a valid model artifact.
- `notebooks/water_pump_analysis.ipynb` is currently empty.
- `app/app.py` imports FastAPI, not Flask. Some older project instructions refer to Flask deployment, but the current application is a FastAPI service.
- The API uses relative model paths, so the working directory matters when starting the server.
- The cleaned dataset has fewer rows than the raw data because the cleaning workflow drops rows with selected missing values.
- The notebooks may contain exploratory cells and package-installation cells; for production use, move dependency installation into `requirements.txt` and keep notebooks focused on analysis.

## Future Improvements

Potential next steps include:

- Populate `requirements.txt` with pinned dependency versions.
- Add automated tests for the API and model input schema.
- Replace relative model paths in `app/app.py` with robust paths based on `pathlib.Path(__file__)`.
- Add a reproducible training script outside the notebooks.
- Track model metrics in a dedicated report or experiment log.
- Add API input validation for known categorical levels.
- Add Docker support for reproducible deployment.
- Add CI checks for linting, tests, and notebook execution.

## License and Data Attribution

Add license and data-source details here before publishing the repository publicly. If this dataset comes from a competition, training programme, or external provider, include the provider name, usage terms, and any citation requirements.
