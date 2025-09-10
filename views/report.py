import streamlit as st
from typing import Dict, List
from utils.chart import get_available_graphs, get_graphs_and_insights
from utils.pdf import create_pdf


def show_report_section(self, columns_info, df):
    st.markdown("---")
    st.subheader("Create Report")
        
    cat_cols = columns_info['categorical']
    numeric_cols = columns_info['numeric'] 
    datetime_cols = columns_info['datetime']
        
    available_graphs = get_available_graphs(cat_cols, numeric_cols, datetime_cols)
        
    selected_graphs = st.multiselect(
        "Select specific graph examples to include in the PDF:",
        options=available_graphs
    )
        
    if selected_graphs:
        report_title = st.text_input("Enter a title for the report:", 
                                       value="Data Analysis Report")
            
        if st.button("Generate PDF Report"):
            self.generate_pdf_report(selected_graphs, df, report_title)




 def generate_pdf_report(self, selected_graphs, df, report_title):
      
    try:
        all_insights, all_graphs = get_graphs_and_insights(selected_graphs, df)
        pdf_data = create_pdf(all_graphs, report_title, all_insights)
            
        with open(pdf_data, "rb") as f:
            pdf_bytes = f.read()
            
        st.success("Report generated!")
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name="selected_graphs_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")
