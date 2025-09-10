import streamlit as st
import pandas as pd



def auto_detect_dates(df):
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df



def apply_filters(df):

   
    if df.empty:
        return df
    
    filtered_df = df.copy()
    original_count = len(df)
    
    # Apply date filters
    filtered_df = apply_date_filters(filtered_df)
    
    # Apply categorical filters  
    filtered_df = apply_categorical_filters(filtered_df)
    
    # Apply numerical filters
    filtered_df = apply_numerical_filters(filtered_df)
    
    # Show filter summary
    filtered_count = len(filtered_df)
    if filtered_count < original_count:
        st.info(f"Filtered: {filtered_count:,} rows (from {original_count:,} original)")
    
    return filtered_df


def apply_date_filters(df):

    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col].dtype)]
    
    if not date_cols:
        return df
    
    st.write("**Date Filters:**")
    filtered_df = df.copy()
    
    for col in date_cols:
        try:
            min_date = df[col].dropna().min().date()
            max_date = df[col].dropna().max().date()
            
            # Skip if all values are null or same date
            if pd.isna(min_date) or min_date == max_date:
                st.write(f"{col}: No date range available")
                continue
                
            date_range = st.date_input(
                f"Filter {col}:",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                key=f"date_{col}"
            )
            
            # Apply filter if valid range selected
            if len(date_range) == 2 and date_range[0] <= date_range[1]:
                start_date, end_date = date_range
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date) 
                
                filtered_df = filtered_df[
                    filtered_df[col].isna() | 
                    ((filtered_df[col] >= start) & (filtered_df[col] <= end))
                ]
            elif len(date_range) == 2:
                st.warning(f"Invalid date range for {col}: start date must be <= end date")
            
        except Exception as e:
            st.error(f"Error processing date column {col}: {str(e)}")
    
    return filtered_df


def apply_categorical_filters(df):
    # Get categorical columns with reasonable number of categories
    cat_cols = []
    for col in df.columns:
        if (df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])):
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 50:  # Skip binary or too many categories
                cat_cols.append(col)
    
    if not cat_cols:
        return df
    
    st.write("**Categorical Filters:**")
    filtered_df = df.copy()
    
    # Use columns for better layout
    num_cols = min(len(cat_cols), 3)
    filter_cols = st.columns(num_cols)
    
    for i, col in enumerate(cat_cols):
        with filter_cols[i % num_cols]:
            try:
                # Get unique values (excluding NaN)
                unique_values = sorted([str(x) for x in df[col].dropna().unique()])
                
                if not unique_values:
                    st.write(f"{col}: No values to filter")
                    continue
                
                # Show filter options
                filter_type = st.radio(
                    f"Filter type for {col}:",
                    ["Include", "Exclude"],
                    key=f"filter_type_{col}",
                    horizontal=True
                )
                
                if filter_type == "Include":
                    selected_values = st.multiselect(
                        f"Include {col}:",
                        options=unique_values,
                        default=unique_values,  # All selected by default
                        key=f"include_{col}"
                    )
                    
                    if selected_values:
                        # Convert back to original dtypes for comparison
                        mask = filtered_df[col].astype(str).isin(selected_values) | filtered_df[col].isna()
                        filtered_df = filtered_df[mask]
                    else:
                        # Nothing selected = empty result
                        filtered_df = filtered_df.iloc[0:0]
                
                else:  # Exclude
                    excluded_values = st.multiselect(
                        f"Exclude {col}:",
                        options=unique_values,
                        default=[],  # Nothing excluded by default
                        key=f"exclude_{col}"
                    )
                    
                    if excluded_values:
                        # Exclude selected values
                        mask = ~filtered_df[col].astype(str).isin(excluded_values) | filtered_df[col].isna()
                        filtered_df = filtered_df[mask]
            
            except Exception as e:
                st.error(f"Error processing categorical column {col}: {str(e)}")
    
    return filtered_df


def apply_numerical_filters(df):

    # Get numerical columns (excluding likely identifiers)
    num_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Skip if likely identifier (high uniqueness)
            if not is_likely_identifier(df[col]):
                num_cols.append(col)
    
    if not num_cols:
        return df
    
    st.write("**Numerical Filters:**")
    filtered_df = df.copy()
    
    # Use columns for better layout
    num_filter_cols = st.columns(min(len(num_cols), 2))
    
    for i, col in enumerate(num_cols):
        with num_filter_cols[i % len(num_filter_cols)]:
            try:
                # Get min/max excluding NaN
                series_clean = df[col].dropna()
                
                if len(series_clean) == 0:
                    st.write(f"{col}: No values to filter")
                    continue
                
                min_val = float(series_clean.min())
                max_val = float(series_clean.max())
                
                # Skip if constant values
                if min_val == max_val:
                    st.write(f"{col}: {min_val:.2f} (constant)")
                    continue
                
                # Determine step size for slider
                range_size = max_val - min_val
                if range_size > 1000:
                    step = 10.0
                elif range_size > 100:
                    step = 1.0
                elif range_size > 10:
                    step = 0.1
                else:
                    step = 0.01
                
                # Range slider
                range_values = st.slider(
                    f"Range for {col}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                    key=f"num_{col}",
                    format="%.2f"
                )
                
                # Apply filter
                if range_values[0] > min_val or range_values[1] < max_val:
                    mask = (
                        ((filtered_df[col] >= range_values[0]) & (filtered_df[col] <= range_values[1])) |
                        filtered_df[col].isna()
                    )
                    filtered_df = filtered_df[mask]
            
            except Exception as e:
                st.error(f"Error processing numerical column {col}: {str(e)}")
    
    return filtered_df


# Simplified setup function (if you want to keep it separate)
def setup_date_filters(df):
    """Legacy function - kept for compatibility"""
    st.warning("⚠️ This function is deprecated. Use apply_date_filters() instead.")
    return {}


def show_column_summary(columns_info):
    # Show excluded columns
    if columns_info['excluded']:
        with st.expander("ℹ️ Columns excluded from visualization"):
            for col, reason in columns_info['excluded']:
                st.write(f"**{col}**: {reason}")
            st.info("These columns were automatically excluded because they appear to be identifiers or have too many unique values for meaningful visualization.")
        
    # Show column summary
    st.write("### Column Summary for Visualization")
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.metric("Numeric Columns", len(columns_info['numeric']))
        if columns_info['numeric']:
            st.write("Available:", ", ".join(columns_info['numeric'][:3]) + 
                    ("..." if len(columns_info['numeric']) > 3 else ""))
        
    with col2:
        st.metric("Categorical Columns", len(columns_info['categorical']))
        if columns_info['categorical']:
            st.write("Available:", ", ".join(columns_info['categorical'][:3]) + 
                        ("..." if len(columns_info['categorical']) > 3 else ""))
        
    with col3:
        st.metric("DateTime Columns", len(columns_info['datetime']))
        if columns_info['datetime']:
            st.write("Available:", ", ".join(columns_info['datetime'][:3]) + 
                    ("..." if len(columns_info['datetime']) > 3 else ""))
