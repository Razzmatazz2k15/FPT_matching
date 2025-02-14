import os
import pandas as pd
import re
import numpy as np
from rapidfuzz import fuzz, process
from user_agents import parse


def filter_by_brand(log_brand: str, device_dictionary: pd.DataFrame, fuzzy_threshold: int = 80) -> pd.DataFrame:
    """
    Filters the device dictionary to match the brand from the log entry.

    Args:
        log_brand (str): The brand from the log entry.
        device_dictionary (pd.DataFrame): The device dictionary DataFrame with a 'Brand' column.
        fuzzy_threshold (int): The minimum similarity score for fuzzy matching.

    Returns:
        pd.DataFrame: Subset of the device dictionary matching the brand.
    """
    if not log_brand or not isinstance(log_brand, str):
        return pd.DataFrame()  # Return empty DataFrame for invalid input

    # Normalize log brand
    log_brand_normalized = log_brand.lower().strip()

    # Step 1: Exact Matching
    exact_match = device_dictionary[
        device_dictionary["Brand"].str.lower().str.strip() == log_brand_normalized
    ]
    if not exact_match.empty:
        return exact_match

    # Step 2: Fuzzy Matching
    # Extract all unique brands from the dictionary
    unique_brands = device_dictionary["Brand"].str.lower().str.strip().unique()

    # Perform fuzzy matching
    match, score, _ = process.extractOne(log_brand_normalized, unique_brands)

    # Check if the best match meets the threshold
    if score >= fuzzy_threshold:
        # Return rows matching the best fuzzy match
        fuzzy_match = device_dictionary[
            device_dictionary["Brand"].str.lower().str.strip() == match
        ]
        return fuzzy_match

    # Step 3: No Match Found
    return pd.DataFrame()  # Return empty DataFrame if no match is found


def filter_by_model_exact(log_model: str, log_marketing_name: str, device_dictionary: pd.DataFrame) -> dict:
    """
    Filters the device dictionary to find an exact match for the given log model or marketing name.

    Args:
        log_model (str): The model from the log entry.
        log_marketing_name (str): The marketing name from the log entry.
        device_dictionary (pd.DataFrame): The device dictionary containing Model Name and Models.

    Returns:
        dict: Best match details or None if no exact match is found.
    """
    # # Normalize input
    # log_model = log_model.lower().strip()
    # log_marketing_name = log_marketing_name.lower().strip()

    # Case 1: Both are "unknown"
    if log_model == "unknown" and log_marketing_name == "unknown":
        return {"matched_model_name": None, "matched_models": None, "source": "No Match"}

    # Case 2: Only one value is "unknown"
    if log_model == "unknown":
        log_model = log_marketing_name
    elif log_marketing_name == "unknown":
        log_marketing_name = log_model

    # Case 3: Check for exact matches
    for search_term, column_name in [
        (log_model, "Model Name"),
        (log_marketing_name, "Model Name"),
        (log_model, "Models"),
        (log_marketing_name, "Models"),
    ]:
        exact_match = device_dictionary[
           device_dictionary[column_name].str.lower().str.strip().str.contains(search_term) #Make contains, therefore this might only work when brand subset is completed
        ]
        if not exact_match.empty:
            return {
                "matched_model_name": exact_match.iloc[0]["Model Name"],
                "matched_models": exact_match.iloc[0]["Models"],
                "source": f"{column_name} (Exact)"
            }

    # No match found
    return {"matched_model_name": None, "matched_models": None, "source": "No Match"}

if __name__ == "__main__":
    # -----------------------------
    # Test filter_by_brand function
    # -----------------------------
    print("Testing filter_by_brand...")

    # Create a dummy device dictionary DataFrame with a 'Brand' column.
    data_brand = {
        "Brand": ["Apple", "Samsung", "Google", "OnePlus"]
    }
    df_brand = pd.DataFrame(data_brand)

    # Test Case 1: Exact match
    log_brand_exact = "Samsung"
    result_exact = filter_by_brand(log_brand_exact, df_brand)
    print("\nExact match result for filter_by_brand (Samsung):")
    print(result_exact)

    # Test Case 2: Fuzzy match (intentional typo)
    log_brand_fuzzy = "samsng"  # slight typo intended for fuzzy matching
    result_fuzzy = filter_by_brand(log_brand_fuzzy, df_brand)
    print("\nFuzzy match result for filter_by_brand (samsng):")
    print(result_fuzzy)

    # Test Case 3: No match (brand not present)
    log_brand_none = "Nokia"
    result_none = filter_by_brand(log_brand_none, df_brand)
    print("\nNo match result for filter_by_brand (Nokia):")
    print(result_none)

    # ------------------------------------------
    # Test filter_by_model_exact function
    # ------------------------------------------
    print("\nTesting filter_by_model_exact...")

    # Create a dummy device dictionary DataFrame for model matching.
    data_model = {
        "Model Name": ["iPhone 12", "Galaxy S21", "Pixel 5", "Nord"],
        "Models": ["A2172, A2402", "SM-G991B", "GD1YQ", "A series"]
    }
    df_model = pd.DataFrame(data_model)

    # Test Case 1: Exact match on "Model Name"
    log_model = "Galaxy S21"
    log_marketing_name = "unknown"
    result_model_exact = filter_by_model_exact(log_model, log_marketing_name, df_model)
    print("\nExact match result for filter_by_model_exact (Galaxy S21):")
    print(result_model_exact)

    # Test Case 2: Both values are "unknown"
    log_model_unknown = "unknown"
    log_marketing_unknown = "unknown"
    result_model_unknown = filter_by_model_exact(log_model_unknown, log_marketing_unknown, df_model)
    print("\nNo match result for filter_by_model_exact (both unknown):")
    print(result_model_unknown)

    # Test Case 3: Contains match on "Models" column
    # For instance, the log entry provides a part of the model identifier.
    log_model_contains = "GD1YQ"
    log_marketing_contains = "unknown"
    result_model_contains = filter_by_model_exact(log_model_contains, log_marketing_contains, df_model)
    print("\nContains match result for filter_by_model_exact (GD1YQ):")
    print(result_model_contains)

    print("\nAll tests completed successfully!")

