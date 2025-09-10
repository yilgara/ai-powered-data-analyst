import streamlit as st
from utils.data_processor import read_file, filter_columns_for_visualization, clean_data
from views.visualization_manager import create_all_visualizations
from views.report import show_report_section
from views.qa_system import show_qa_section
from views.data import apply_filters, auto_detect_dates, show_column_summary


def main():
    st.title("AI Data Analysis & PDF Report Generator")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        # Process data
        df = read_file(uploaded_file)
        df = auto_detect_dates(df)

        # Clean data
        cleaned_df = clean_data(df)
        
        # Apply filters
        filtered_df = apply_filters(cleaned_df)
        
        
        st.write("### Filtered & Cleaned Data")
        st.dataframe(filtered_df)


        # Get suitable columns
        columns_info = filter_columns_for_visualization(filtered_df)
        show_column_summary(columns_info)


        # Generate visualizations
        create_all_visualizations(filtered_df, columns_info)
        
        # Report generation section
        show_report_section(columns_info, filtered_df)
        
        # Q&A section
        show_qa_section(filtered_df)
        

if __name__ == "__main__":
    main()
