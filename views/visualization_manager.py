import streamlit as st
import seaborn as sns
from utils.chart import *


def create_all_visualizations(df, columns_info):
    st.write("### Charts for filtered and cleaned data")
        
    if df.empty:
        st.warning("No data available for visualization")
        return 
        
    numeric_cols = columns_info['numeric']
    cat_cols = columns_info['categorical']
    datetime_cols = columns_info['datetime']

    # Create individual visualizations
    create_histogram(df, numeric_cols)
    create_time_series(df, datetime_cols, numeric_cols)
    create_bar_chart(df, cat_cols)
    create_num_vs_cat_analysis(df, cat_cols, numeric_cols)
    create_cat_vs_cat_analysis(df, cat_cols)
    create_num_vs_num_analysis(df, numeric_cols)
        

def add_visualization(fig, insight):
    all_figures.append(fig)
    st.pyplot(fig)
    for i in insight:
        st.info(i)


def create_histogram(df, numeric_cols):
    if numeric_cols:
        st.subheader("Numerical Analysis - Histogram")
        selected_col = st.selectbox("Select a numeric column for the histogram", numeric_cols)
        fig, insight = plot_histogram(df, selected_col)
        add_visualization(fig, insight)

def create_time_series(df, datetime_cols, numeric_cols):
    if datetime_cols and numeric_cols:
        st.subheader("Time Series Chart")
        dt_col = st.selectbox("Date/Time column", datetime_cols)
        num_col = st.selectbox("Numeric column", numeric_cols, key="ts_plot")
        fig, insight = plot_time_series(df, dt_col, num_col)
        add_visualization(fig, insight)

def create_bar_chart(df, cat_cols):
    if cat_cols:
        st.subheader("Categorical Analysis - Bar Plot")
        cat_col = st.selectbox("Select a categorical column for the bar chart", cat_cols)
        fig, insight = plot_bar_chart(df, cat_col)
        self._add_visualization(fig, insight)


def create_num_vs_cat_analysis(df, cat_cols, numeric_cols):
   
    if cat_cols and numeric_cols:
        st.subheader("Numerical vs Categorical Analysis")
            
        x_cat = st.selectbox("Select a categorical column", cat_cols)
        y_num = st.selectbox("Select a numeric column", numeric_cols)
            
        plot_type = st.radio("Select plot type:", ["Box Plot", "Violin Plot", "Bar Plot"], horizontal=True)
            
        if plot_type == "Box Plot":
            fig, insight = plot_boxplot(df, y_num, x_cat)
        elif plot_type == "Violin Plot":
            fig, insight = plot_violinplot(df, y_num, x_cat)
        elif plot_type == "Bar Plot":
            fig, insight = plot_barplot_num_vs_cat(df, y_num, x_cat)
            
         add_visualization(fig, insight)


def create_cat_vs_cat_analysis(df, cat_cols):
    if len(cat_cols) >= 2:
        st.subheader("Categorical vs Categorical Analysis")
        x_cat = st.selectbox("Select a categorical column (X-axis)", cat_cols, key="cat_vs_cat_x")
        y_cat = st.selectbox("Select a categorical column (Y-axis)", 
                            [col for col in cat_cols if col != x_cat], key="cat_vs_cat_y")
            
        if x_cat and y_cat:
            chart_type = st.radio("Select chart type:", ["Stacked Bar", "Heatmap"], horizontal=True)
                
            if chart_type == "Stacked Bar":
                fig, insight = plot_stacked_bar(df, x_cat, y_cat)
            else:  # Heatmap
                fig, insight = plot_heat_map(df, x_cat, y_cat)
                
             add_visualization(fig, insight)


def create_num_vs_num_analysis(df, numeric_cols):
    if len(numeric_cols) >= 2:
        st.subheader("Numerical vs Numerical Analysis")
        x_num = st.selectbox("Select a numerical column (X-axis)", numeric_cols, key="num_vs_num_x")
        y_num = st.selectbox("Select a numerical column (Y-axis)", 
                            [col for col in numeric_cols if col != x_num], key="num_vs_num_y")
            
        if x_num and y_num:
            fig, insight, ax = plot_scatter(df, x_num, y_num)
                
            add_reg = st.checkbox("Add regression line", value=False)
            if add_reg:
                sns.regplot(data=df, x=x_num, y=y_num, scatter=False, ax=ax, color='red')
                
            add_visualization(fig, insight)
 
