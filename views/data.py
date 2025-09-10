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
    setup_date_filters(df)
        
    filtered_df = df.copy()
    for col, (start_date, end_date) in self.filters.items():
        if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            filtered_df = filtered_df[
                filtered_df[col].isna() | 
                ((filtered_df[col] >= start) & (filtered_df[col] <= end))
            ]
        
    return filtered_df


def setup_date_filters(df):
    self.filters = {}
        
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
            min_date = df[col].min().date()
            max_date = df[col].max().date()
            date_range = st.date_input(
                f"Filter according to the {col} column", 
                value=[min_date, max_date]
            )
                
            if len(date_range) == 2:
                start_date, end_date = date_range
                if start_date != end_date:
                    self.filters[col] = (start_date, end_date)
                else:
                    st.warning("Please select a different end date for a valid range.")
            else:
                st.warning("Please select both a start and an end date.")


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
