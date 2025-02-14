import os
import re
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

#_______________________________ Data Cleaning __________________________________________

def identify_categorical_columns(df:  pd.DataFrame):
    """
    Identifies which columns in a DataFrame have a categorical constraint.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: A list of column names with categorical constraints.
    """
    # Use Pandas select_dtypes to find columns of type 'category'
    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()

    return categorical_columns

def proccess_online_web(df:  pd.DataFrame):
  '''
  Proccessing online web
  Should take uncleaned online web data
  Returns clean dataset to use.
  '''
  important_columns = [ #Take the important colums
        "cdp_id",
        "user_id",
        "device_brand",
        "device_model",
        "device_type",
        "os",
        "os_version",
        "user_agent",
        "context_device_model",
        "marketing_name"
    ]
  df_filtered = df[important_columns].copy()

  #Standarize text columns
  for col in ["device_brand", "device_model", "device_type", "os",
              "context_device_model", "marketing_name"]:
      if col in df_filtered.columns:
          df_filtered[col] = df_filtered[col].fillna("unknown").str.lower().str.strip().copy()

  #Fill missing user_agent and user_id
  df_filtered["user_agent"] = df_filtered["user_agent"].fillna("unknown")
  df_filtered["user_id"] = df_filtered["user_id"].fillna(-1).astype(int).astype(str)

  return df_filtered

def log_data_clean(df : pd.DataFrame):
    """
    Cleans a dataset by:
    - Replacing NaN values with "unknown".
    - Converting all text columns to lowercase, except for 'user_agent' and 'mac'.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: A new cleaned dataframe (original remains unchanged).
    """

    # Make a copy to avoid modifying the original dataframe1
    df_cleaned = df.copy()

    # Replace NaN values with "unknown"
    df_cleaned.fillna("unknown", inplace=True)

    # List of columns to exclude
    exclude_columns = ['user_agent', 'mac']

    # Loop through all columns and convert to lowercase if not in exclude list
    for col in df_cleaned.columns:
        if col not in exclude_columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.lower()

    return df_cleaned

def procces_online_app(df:  pd.DataFrame):
    important_columns = [
      "user_agent",
      "brand",
      "model",
      "marketing_name",
      "os",
      "device_type",
      "is_mobile",
      "device_id"
    ]

    df_filtered = df[important_columns].copy()

    for col in ["brand", "model", "marketing_name", "os", "device_type", "device_id"]:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].fillna("unknown").str.lower().str.strip()

    df_filtered["user_agent"] = df_filtered["user_agent"].fillna("unknown")
    
    return df_filtered

def process_hifpt(df : pd.DataFrame):
  '''
  Proccesing FPT HiFPT dataset
  '''
  important_columns = [
    "customer_id",
    "device_id",
    "device_name",
    "mac",
    "device_model",
    "brand",
    "marketing_name",
    "map_key"]

  df_filtered = df[important_columns].copy()

  for col in ["device_name", "device_model", "brand", "marketing_name", "mac", "map_key"]:
    if col in df_filtered.columns:
        df_filtered[col] = (
            df_filtered[col]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .str.strip())

  #Might have to change Nan values to unknown for string compatibility
  return df_filtered

def proccess_telecom(df :  pd.DataFrame):
  '''
  Proccess the FPT telecom dataset.
  Takes the ubncleaned dataset as input and returns cleaned version
  '''
  important_columns = [
    "mac_device",
    "device_type",
    "device_name",
    "vendor_short",
    "vendor",
    "source"
  ]

  df_filtered = df[important_columns].copy()

  for col in ["device_name", "vendor_short", "vendor", "source"]:
    if col in df_filtered.columns:
        df_filtered[col] = (
            df_filtered[col]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .str.strip()
            )

  return df_filtered

# ______________________________ AI PROCESSSING __________________________________________

def load_sentence_transformer(model_name: str, trust_remote_code: bool = False) -> SentenceTransformer:
    """
    Loads a SentenceTransformer model for a given model identifier.
    
    Parameters:
        model_name (str): The SentenceTransformer model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2").
        trust_remote_code (bool): Whether to trust and execute remote code from the model repository.
        
    Returns:
        SentenceTransformer: The loaded model with integrated tokenizer and pooling.
    """
    try:
        model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    except Exception as e:
        raise RuntimeError(f"Error loading SentenceTransformer model '{model_name}': {e}")
    return model

def precompute_column_embeddings(df: pd.DataFrame, column: str, model: SentenceTransformer, batch_size: int = 32) -> torch.Tensor:
    """
    Precompute embeddings for a specific text column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing your device dictionary.
        column (str): The name of the column to encode (e.g., "brand" or "model").
        model (SentenceTransformer): A pre-loaded SentenceTransformer model.
        batch_size (int): Number of texts to process per batch.
        
    Returns:
        torch.Tensor: A tensor of shape (num_rows, embedding_dim) containing the embeddings for the given column.
    """
    # Convert the column to strings (in case there are non-string values) and create a list.
    texts = df[column].astype(str).tolist()
    # Use the model's encode() method to compute embeddings efficiently in batches.
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True)
    return embeddings

def match_all_queries( embeddings_sample: torch.Tensor, corpus_embeddings: torch.Tensor, corpus_texts: List[str],
    sample_texts: List[str],
    threshold: float = 0.90
) -> Tuple[List[dict], List[dict]]:
    """
    Computes cosine similarity between all sample embeddings and corpus embeddings in one matrix operation.
    
    Parameters:
        embeddings_sample (torch.Tensor): Tensor of shape (N, d) for N log entry embeddings.
        corpus_embeddings (torch.Tensor): Tensor of shape (M, d) for M device dictionary embeddings.
        corpus_texts (List[str]): List of M original texts corresponding to corpus_embeddings.
        sample_texts (List[str]): List of N original log entry texts corresponding to embeddings_sample.
        threshold (float): Similarity threshold for a match.
        
    Returns:
        Tuple[List[dict], List[dict]]: Two lists containing the matching results and non-matching results.
            Each result is a dict with keys:
                "query_index": index of the sample,
                "sample_text": the original sample text,
                "matched_text": the dictionary entry that is the best match,
                "score": cosine similarity score.
    """
    # Normalize both sets of embeddings
    embeddings_sample = torch.nn.functional.normalize(embeddings_sample, p=2, dim=1)
    corpus_embeddings = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix (shape: N x M)
    cosine_matrix = util.cos_sim(embeddings_sample, corpus_embeddings)
    
    # For each sample, find the best matching index and corresponding score
    best_scores, best_indices = torch.max(cosine_matrix, dim=1)
    
    match_list = []
    non_match_list = []
    
    for i, (score, idx) in enumerate(zip(best_scores, best_indices)):
        result = {
            "query_index": i,
            "sample_text": sample_texts[i],
            "matched_text": corpus_texts[int(idx)],
            "score": score.item()
        }
        if score.item() >= threshold:
            match_list.append(result)
        else:
            non_match_list.append(result)
    
    return match_list, non_match_list

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    print("Testing identify_categorical_columns...")
    # Create a DataFrame with a categorical column.
    df_cat = pd.DataFrame({
        "a": pd.Categorical(["x", "y", "z"]),
        "b": [1, 2, 3]
    })
    cats = identify_categorical_columns(df_cat)
    print("Categorical columns:", cats)

    print("\nTesting proccess_online_web...")
    # Create a dummy DataFrame with the required columns.
    df_online_web = pd.DataFrame({
        "cdp_id": [1, 2],
        "user_id": [None, 2],
        "device_brand": ["Samsung", None],
        "device_model": ["Galaxy", "Note"],
        "device_type": ["phone", "tablet"],
        "os": ["Android", "iOS"],
        "os_version": ["10", "14"],
        "user_agent": [None, "Mozilla"],
        "context_device_model": ["SM-G950F", "iPhone"],
        "marketing_name": [None, "Galaxy S10"]
    })
    online_web_clean = proccess_online_web(df_online_web)
    print(online_web_clean.head())

    print("\nTesting log_data_clean...")
    # Create a dummy DataFrame with some NaN and uppercase text.
    df_log = pd.DataFrame({
        "col1": ["HELLO", None],
        "user_agent": ["Chrome", "Firefox"],
        "mac": [None, "00:1A:2B:3C:4D:5E"]
    })
    log_clean = log_data_clean(df_log)
    print(log_clean.head())

    print("\nTesting procces_online_app...")
    # Create a dummy DataFrame with the required columns.
    df_online_app = pd.DataFrame({
        "user_agent": [None, "Safari"],
        "brand": ["Apple", "Samsung"],
        "model": ["iPhone", "Galaxy"],
        "marketing_name": ["iPhone 12", "Galaxy S21"],
        "os": ["iOS", "Android"],
        "device_type": ["phone", "phone"],
        "is_mobile": [True, True],
        "device_id": ["id1", "id2"]
    })
    online_app_clean = procces_online_app(df_online_app)
    print(online_app_clean.head())

    print("\nTesting process_hifpt...")
    # Create a dummy DataFrame for HiFPT processing.
    df_hifpt = pd.DataFrame({
        "customer_id": [1, 2],
        "device_id": ["d1", "d2"],
        "device_name": ["DeviceA", None],
        "mac": [None, "00:AA:BB:CC:DD:EE"],
        "device_model": ["ModelX", "ModelY"],
        "brand": ["BrandX", None],
        "marketing_name": [None, "MarketingY"],
        "map_key": ["key1", "key2"]
    })
    hifpt_clean = process_hifpt(df_hifpt)
    print(hifpt_clean.head())

    print("\nTesting proccess_telecom...")
    # Create a dummy DataFrame for telecom processing.
    df_telecom = pd.DataFrame({
        "mac_device": ["00:11:22:33:44:55", "66:77:88:99:AA:BB"],
        "device_type": ["router", "modem"],
        "device_name": ["Device1", None],
        "vendor_short": ["VS1", "VS2"],
        "vendor": ["Vendor1", None],
        "source": [None, "Source2"]
    })
    telecom_clean = proccess_telecom(df_telecom)
    print(telecom_clean.head())

    print("\nTesting load_sentence_transformer and precompute_column_embeddings...")
    try:
        # Load a SentenceTransformer model.
        model = load_sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")
        # Create a small DataFrame with text.
        df_text = pd.DataFrame({"text": ["This is a test.", "Another test."]})
        embeddings = precompute_column_embeddings(df_text, "text", model, batch_size=2)
        print("Embeddings shape:", embeddings.shape)
    except Exception as e:
        print("Error in loading model or computing embeddings:", e)

    print("\nTesting match_all_queries...")
    # Create dummy embeddings and texts.
    # For simplicity, we'll use 3-dimensional vectors.
    sample_embeddings = torch.tensor([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]])
    corpus_embeddings = torch.tensor([[1.0, 0.1, 0.0],
                                        [0.0, 1.0, 0.1],
                                        [0.1, 0.0, 1.0]])
    sample_texts = ["sample1", "sample2"]
    corpus_texts = ["corpus1", "corpus2", "corpus3"]
    matches, non_matches = match_all_queries(sample_embeddings, corpus_embeddings, corpus_texts, sample_texts, threshold=0.9)
    print("Matches:", matches)
    print("Non-matches:", non_matches)

    print("\nAll tests completed successfully!")

    

