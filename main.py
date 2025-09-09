
from utils.chart import *
from utils.prompt import *
from utils.pdf import create_pdf


def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format: Only .csv and .xlsx are supported.")



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







def main():
    all_figures = []
    all_insights = []

    st.title("AI Data Analysis & PDF Report Generator")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = read_file(uploaded_file)
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        st.write("### Original Data")
        st.dataframe(df)

        filters = {}

        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                min_date = df[col].min().date()
                max_date = df[col].max().date()
                date_range = st.date_input(
                    f"Filter according to the {col} column", value=[min_date, max_date]
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    if start_date == end_date:
                        st.warning("Please select a different end date for a valid range.")
                    else:
                        filters[col] = (start_date, end_date)
                else:
                    st.warning("Please select both a start and an end date.")



        # Apply filters
        filtered_df = df.copy()

        for col, (start_date, end_date) in filters.items():
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)

                filtered_df = filtered_df[
                    filtered_df[col].isna() | ((filtered_df[col] >= start) & (filtered_df[col] <= end))
                    ]


        st.write("### Filtered Data")
        st.dataframe(filtered_df)

        st.write("### Cleaned Data")
        cleaned_df = filtered_df.copy()
        cleaned_df = clean_data(cleaned_df)
        st.dataframe(cleaned_df)


        st.write("### Charts for filtered and cleaned data")

        if not cleaned_df.empty:
            # Example 1: Histogram for numeric column
            numeric_cols = cleaned_df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                st.subheader("Numerical Analysis - Histogram")
                selected_col = st.selectbox("Select a numeric column for the histogram", numeric_cols)
                fig1, insight1 = plot_histogram(cleaned_df, selected_col)
                all_figures.append(fig1)
                st.pyplot(fig1)
                all_insights.append(insight1)
                for i in insight1:
                    st.info(i)

            # Example 2: Time Series Plot for datetime column
            datetime_cols = cleaned_df.select_dtypes(include='datetime').columns.tolist()
            if datetime_cols and numeric_cols:
                st.subheader("Time Series Chart")
                dt_col = st.selectbox("Date/Time column", datetime_cols)
                num_col = st.selectbox("Numeric column", numeric_cols, key="ts_plot")
                fig2, insight2 = plot_time_series(cleaned_df, dt_col, num_col)
                all_figures.append(fig2)
                st.pyplot(fig2)
                all_insights.append(insight2)
                for i in insight2:
                    st.info(i)

            # Example 3: Bar Plot for category counts
            cat_cols = cleaned_df.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                st.subheader("Categorical Analysis - Bar Plot")
                cat_col = st.selectbox("Select a categorical column for the bar chart", cat_cols)
                fig3, insight3 = plot_bar_chart(cleaned_df, cat_col)
                all_figures.append(fig3)
                st.pyplot(fig3)
                all_insights.append(insight3)
                for i in insight3:
                    st.info(i)


            if cat_cols and numeric_cols:
                sst.subheader("Numerical vs Categorical Analysis")

                x_cat = st.selectbox("Select a categorical column", cat_cols)
                y_num = st.selectbox("Select a numeric column", numeric_cols)

                plot_type = st.radio("Select plot type:", ["Box Plot", "Violin Plot", "Bar Plot"], horizontal=True)

                if plot_type == "Box Plot":
                    fig4, insight4 = plot_boxplot(cleaned_df, y_num, x_cat)
                elif plot_type == "Violin Plot":
                    fig4, insight4 = plot_violinplot(cleaned_df, y_num, x_cat)
                elif plot_type == "Bar Plot":
                    fig4, insight4 = plot_barplot_num_vs_cat(cleaned_df, y_num, x_cat)


                st.pyplot(fig4)
                all_insights.append(insight4)
                all_figures.append(fig4)
                for i in insight4:
                    st.info(i)


            if len(cat_cols) >= 2:
                st.subheader("Categorical vs Categorical Analysis")
                x_cat = st.selectbox("Select a categorical column (X-axis)", cat_cols, key="cat_vs_cat_x")
                y_cat = st.selectbox("Select a categorical column (Y-axis)", [col for col in cat_cols if col != x_cat], key="cat_vs_cat_y")
                
                if x_cat and y_cat:

                    chart_type = st.radio("Select chart type:", ["Stacked Bar", "Heatmap"], horizontal=True)

                    if chart_type == "Stacked Bar":
                        fig5, insight5 = plot_stacked_bar(cleaned_df, x_cat, y_cat)

                    else:  # Heatmap
                        fig5, insight5 = plot_heat_map(cleaned_df, x_cat, y_cat)


                    st.pyplot(fig5)
                    all_insights.append(insight5)
                    all_figures.append(fig5)
                    for i in insight5:
                        st.info(i)



            if len(numeric_cols) >= 2:
                st.subheader("Numerical vs Numerical Analysis")
                x_num = st.selectbox("Select a numerical column (X-axis)", numeric_cols, key="num_vs_num_x")
                y_num = st.selectbox("Select a numerical column (Y-axis)", [col for col in numeric_cols if col != x_num], key="num_vs_num_y")

                if x_num and y_num:
                    fig6, insight6, ax = plot_scatter(cleaned_df, x_num, y_num)


                    add_reg = st.checkbox("Add regression line", value=False)
                    
                    if add_reg:
                        sns.regplot(data=cleaned_df, x=x_num, y=y_num, scatter=False, ax=ax, color='red')

                    st.pyplot(fig6)

                    all_figures.append(fig6)
                    all_insights.append(insight6)
                    for i in insight6:
                        st.info(i)

        st.markdown("---")
        st.subheader("Create Report")

        available_graphs = get_available_graphs(cat_cols, numeric_cols, datetime_cols)

        selected_graphs = st.multiselect(
            "Select specific graph examples to include in the PDF:",
            options=available_graphs
        )

        if selected_graphs:
            report_title = st.text_input("Enter a title for the report:", value="Data Analiz Hesabatı")

            if st.button("Generate PDF Report"):

                all_insights, all_graphs = get_graphs_and_insights(selected_graphs, cleaned_df)
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

        st.markdown("---")
        st.subheader("Ask a few questions (GPT will answer)")
        if "question_count" not in st.session_state:
            st.session_state.question_count = 1

        # Button to add a new question section
        if st.button("+ New Question"):
            st.session_state.question_count += 1

        # Collect all questions from user
        questions = []
        for i in range(st.session_state.question_count):
            question = st.text_input(f"Sual {i + 1}:", key=f"q_{i}")
            if question.strip():
                questions.append(question)


        if st.button("Get Answer"):
            if questions:
                combined_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
                with st.spinner("Preparing GPT response..."):
                    gpt_answer = get_response(combined_questions, cleaned_df)
                    answers = parse_gpt_response(gpt_answer, cleaned_df)
                    st.session_state["last_answers"] = answers


            else:
                st.warning("Please enter a question.")

        if "last_answers" in st.session_state:
            st.subheader("Answers:")
            for question, answer in st.session_state["last_answers"].items():
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                st.markdown("---")

       




if __name__ == "__main__":
    main()
