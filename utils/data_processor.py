import pandas as pd


def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format: Only .csv, .xls and .xlsx are supported.")


def is_likely_identifier(series, max_unique_ratio=0.7, min_length=2):
    total_count = len(series.dropna())
    if total_count == 0:
        return False
    
    unique_count = series.nunique()
    unique_ratio = unique_count / total_count
    
    
    # High uniqueness ratio suggests it's an identifier
    return unique_ratio >= max_unique_ratio:



def filter_columns_for_visualization(df, max_categorical_unique=20, max_unique_ratio=0.7):
   
    suitable_numeric = []
    suitable_categorical = []
    suitable_datetime = []
    
    excluded_columns = []
    
    
    
    for col in df.columns:
        dtype = df[col].dtype
        
        # Check if it's likely an identifier
        if is_likely_identifier(df[col], max_unique_ratio):
            excluded_columns.append((col, "High uniqueness - likely identifier"))
            continue
        
        if pd.api.types.is_datetime64_any_dtype(dtype):
            suitable_datetime.append(col)
        
        elif pd.api.types.is_numeric_dtype(dtype):
            suitable_numeric.append(col)
        
        elif dtype == 'object' or pd.api.types.is_string_dtype(dtype):
            unique_count = df[col].nunique()
            suitable_categorical.append(col)

    return {
            'numeric': suitable_numeric,
            'categorical': suitable_categorical,
            'datetime': suitable_datetime,
            'excluded': excluded_columns
        }


def clean_data(df, row_null_threshold=0.5):

    # 1. Drop rows with too many nulls (e.g., more than 50% null)
    df = df.dropna(thresh=int((1 - row_null_threshold) * len(df.columns)))

    # 2. Fill remaining missing values based on data type
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        elif df[col].dtype == 'object':
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    # 3. Drop constant columns (columns with only one unique value)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)

    return df
      
