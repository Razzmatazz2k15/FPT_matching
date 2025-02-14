# Device Matching & Data Processing Module

This repository contains a collection of functions for data cleaning, device matching, and file search operations. It also includes transformer-based functions to load models and compute text embeddings.

## Features

- **Data Cleaning:**  
  - Identify categorical columns in a DataFrame.
  - Clean online web, mobile app, HiFPT, and telecom datasets.
  - Standardize text, handle missing values, and more.

- **Device Matching:**  
  - Filter device dictionaries by brand using both exact and fuzzy matching.
  - Match device models using exact and substring-based methods.

- **File Search Functions:**  
  - Search for files and Parquet directories within a data folder.

- **Transformer Functions:**  
  - Load SentenceTransformer models.
  - Compute text embeddings.
  - Perform query matching based on cosine similarity.

## Requirements

See the `requirements.txt` for a full list of dependencies. If you need to install them manually, run:

```bash
pip install -r requirements.txt
