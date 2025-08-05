import google.generativeai as genai
import pandas as pd
from google.generativeai.types import GenerationConfig
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import re
import streamlit as st
from datetime import datetime
from fpdf.enums import XPos, YPos

import os
import requests

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def download_dejavu_font():
    url = "https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip"
    zip_path = "freefont.zip"
    font_folder = "freefont-20120503"
    font_file = "FreeSans.ttf"

    if not os.path.exists(font_file):
        # Download the zip file if not exists
        if not os.path.exists(zip_path):
            r = requests.get(url)
            if r.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(r.content)
            else:
                raise Exception("Failed to download font zip")

        # Extract the zip and get font file
        import zipfile
        import shutil
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
        extracted_font_path = os.path.join(font_folder, font_file)
        if os.path.exists(extracted_font_path):
            shutil.move(extracted_font_path, font_file)
        else:
            raise FileNotFoundError(f"{extracted_font_path} not found")
    return font_file

def read_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format: Only .csv and .xlsx are supported.")



def build_prompt(df, user_question):
    summary = df.describe(include='all').fillna("").to_markdown()
    types = df.dtypes.to_string()
    missing = df.isnull().sum().to_string()
    sample = df.head(10).to_markdown(index=False)
    return f"""
        You're a Python data analyst.  I have a filtered dataset (already narrowed down by user-selected period and cleaned) with this summary:
        
        **Column Types:**
        {types}
        
        **Missing Values Count:**
        {missing}
        
        **Sample Rows (Top 10):**
        {sample}
        
        **Summary Stats:**
        {summary}
        
        Please:
        
        Answer user questions:
        {user_question}
        
        Key points:
        1. Assume that the DataFrame is named cleaned_df and use it to answer any questions that are related to the dataset.
        2. Store all answers in a single Python dictionary called 'answers'.
            - Each key should be the original user question.
            - Each value should be the corresponding answer (either a Python expression or a string, depending on the case).
        3. If a question is not related to the dataset, do not write executable Python code.
            - Instead, store a plain string response as the value in the 'answers' dictionary, where the key is the original question and the value is the corresponding answer

        Respond ONLY with Python code blocks.
        Important: Please write all **strings, key and value pairs in dictionary** in **Azerbaijani language**, but keep all variable names (like `answers`) in English.
    """





def get_response(user_questions, df, model_name= 'gemini-1.5-flash'):
    prompt = build_prompt(df, user_questions)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are an AI assistant for data analysis."
    )

    generation_config = GenerationConfig(
        temperature=0.3,  # Controls randomness. Lower values are more deterministic.
    )

    try:
        # Send the prompt to the Gemini model
        response = model.generate_content(
            contents=[
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            generation_config=generation_config
        )

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            print("Error: Model did not return expected content structure.")
            return "Content generation failed."

    except Exception as e:
        print(f"An error occurred during content generation: {e}")
        return f"Error: {e}"


def parse_gpt_response(response, df):
    # Extract Python code blocks
    code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
    namespace = {'df': df.copy(), 'plt': plt, 'pd': pd}
    for block in code_blocks:
        try:
            exec(block, namespace)
        except Exception as e:
            print(f"Error running GPT code:\n{e}\n\nCode:\n{block}")

    # Return generated titles, insights, descriptions if available
    answers = namespace.get('answers', {})

    return answers




# Bar chart for categorical column
def plot_bar_chart(df, col):
    fig, ax = plt.subplots()
    value_counts = df[col].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
    ax.set_title(f"{col} kateqoriyasının tezliyi")
    ax.set_xlabel(col)
    ax.set_ylabel("Say")
    plt.xticks(rotation=45)

    insight = [f"Ən çox görülən kateqoriya: '{value_counts.idxmax()}'({value_counts.max()} dəfə).",
               f"Ən az görülən kateqoriya: '{value_counts.idxmin()}'({value_counts.min()} dəfə)."]
    return fig, insight


# Histogram for numeric column
def plot_histogram(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f"{col} sütununun histogramı")
    ax.set_xlabel(col)
    ax.set_ylabel("Say")
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()

    insight = [f"Ortalama: {mean:.2f}", f"Median: {median:.2f}", f"Standart Deviation: {std:.2f}"]
    if mean > median:
        insight.append("Dağılım sağa meyllidir (right-skewed).")
    elif mean < median:
        insight.append("Dağılım sola meyllidir (left-skewed).")
    else:
        insight.append("Dağılım təxmini simmetrikdir.")
    return fig, insight


# Time Series plot
def plot_time_series(df, dt_col, num_col):
    fig, ax = plt.subplots()
    ts_df = df[[dt_col, num_col]].dropna().sort_values(dt_col)

    ts_df[dt_col] = pd.to_datetime(ts_df[dt_col])
    ax.plot(ts_df[dt_col], ts_df[num_col])

    ax.set_title(f"{num_col} dəyişməsi zaman üzrə")
    ax.set_xlabel(dt_col)
    ax.set_ylabel(num_col)

    trend = ts_df[num_col].diff().mean()
    max_idx = ts_df[num_col].idxmax()
    min_idx = ts_df[num_col].idxmin()

    max_val = ts_df.loc[max_idx, num_col]
    min_val = ts_df.loc[min_idx, num_col]

    max_time = ts_df.loc[max_idx, dt_col].strftime('%Y-%m-%d')
    min_time = ts_df.loc[min_idx, dt_col].strftime('%Y-%m-%d')
    fig.autofmt_xdate()

    direction = f"{num_col} zamanla artma meyli göstərir." if trend > 0 else f"{num_col} zamanla azalma meyli göstərir." if trend < 0 else f"{num_col} zamanla sabit qalır."
    insight = [direction, f"{num_col} maksimum {max_val} dəyərinə {max_time} tarixində çatdı.",
               f"{num_col} minimum {min_val} dəyərinə {min_time} tarixində çatdı."]
    return fig, insight


# Stacked Bar for two categorical columns
def plot_stacked_bar(df, cat1, cat2):
    fig, ax = plt.subplots()
    ctab = pd.crosstab(df[cat1], df[cat2])
    ctab.plot(kind='bar', stacked=True, ax=ax)

    ax.set_title(f"{cat1} və {cat2} üzrə Stacked Bar Chart")
    ax.set_xlabel(cat1)
    ax.set_ylabel(cat2)

    max_val = ctab.stack().max()
    min_val = ctab.stack().min()

    most_common = ctab.stack()[ctab.stack() == max_val].index.tolist()
    least_common = ctab.stack()[ctab.stack() == min_val].index.tolist()

    most_common_str = ", ".join([f"{x} & {y}" for x, y in most_common])
    most_comb = f"Ən çox yayılmış kombinasiyalar: {most_common_str} ({max_val} dəfə)."

    least_common_str = ", ".join([f"{x} & {y}" for x, y in least_common])
    least_comb = f"Ən az yayılmış kombinasiyalar: {least_common_str} ({min_val} dəfə)."

    insight = [most_comb, least_comb]

    return fig, insight


def plot_heat_map(df, cat1, cat2):
    ctab = pd.crosstab(df[cat1], df[cat2])

    fig, ax = plt.subplots()
    sns.heatmap(ctab, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_title(f"{cat1} və {cat2} üzrə Heat Map")
    ax.set_xlabel(cat1)
    ax.set_ylabel(cat2)

    max_val = ctab.stack().max()
    min_val = ctab.stack().min()

    most_common = ctab.stack()[ctab.stack() == max_val].index.tolist()
    least_common = ctab.stack()[ctab.stack() == min_val].index.tolist()

    most_common_str = ", ".join([f"{x} & {y}" for x, y in most_common])
    most_comb = f"Ən çox yayılmış kombinasiyalar: {most_common_str} ({max_val} dəfə)."

    least_common_str = ", ".join([f"{x} & {y}" for x, y in least_common])
    least_comb = f"Ən az yayılmış kombinasiyalar: {least_common_str} ({min_val} dəfə)."

    insight = [most_comb, least_comb]
    return fig, insight


# Scatter plot for two numeric columns
def plot_scatter(df, num1, num2):
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[num1], y=df[num2], ax=ax)
    ax.set_title(f"{num1} və {num2} üzrə Scatter Plot")
    ax.set_xlabel(num1)
    ax.set_ylabel(num2)

    corr = df[[num1, num2]].corr().iloc[0, 1]
    insight = [f"{num1} ilə {num2} arasında korrelyasiya: {corr:.2f}."]

    if abs(corr) > 0.7:
        insight.append("Bu sütunlar arasında güclü əlaqə var.")
    elif abs(corr) > 0.4:
        insight.append("Orta səviyyəli korrelyasiya mövcuddur.")
    else:
        insight.append("Əlaqə zəifdir və ya mövcud deyil.")
    return fig, insight, ax


# Box plot for numeric vs categorical
def plot_boxplot(df, num, cat):
    fig, ax = plt.subplots()
    sns.boxplot(x=cat, y=num, data=df, ax=ax)
    ax.set_title(f"{num} dağılımı üzrə {cat}")
    plt.xticks(rotation=45)

    group_means = df.groupby(cat)[num].mean()
    most = group_means.idxmax()
    least = group_means.idxmin()

    insight = [
        f"{most} kateqoriyası ən yüksək orta {num} dəyərinə malikdir ({group_means[most]:.2f}).",
        f"{least} kateqoriyası ən aşağı orta {num} dəyərinə malikdir ({group_means[least]:.2f})."
    ]
    return fig, insight


# Violin plot for numeric vs categorical
def plot_violinplot(df, num, cat):
    fig, ax = plt.subplots()
    sns.violinplot(x=cat, y=num, data=df, ax=ax)
    ax.set_title(f"{num} sıxlığı üzrə {cat}")
    plt.xticks(rotation=45)

    group_means = df.groupby(cat)[num].mean()
    most = group_means.idxmax()
    least = group_means.idxmin()

    insight = [
        f"{most} kateqoriyası ən yüksək orta {num} dəyərinə malikdir ({group_means[most]:.2f}).",
        f"{least} kateqoriyası ən aşağı orta {num} dəyərinə malikdir ({group_means[least]:.2f})."
    ]
    return fig, insight


# Bar plot for numeric vs categorical (mean bar)
def plot_barplot_num_vs_cat(df, num, cat):
    fig, ax = plt.subplots()
    sns.barplot(x=df[cat], y=df[num], ax=ax, estimator="mean")
    ax.set_title(f"{cat} üzrə {num} orta dəyəri")

    group_means = df.groupby(cat)[num].mean()
    most = group_means.idxmax()
    least = group_means.idxmin()

    insight = [
        f"{most} kateqoriyası ən yüksək orta {num} dəyərinə malikdir ({group_means[most]:.2f}).",
        f"{least} kateqoriyası ən aşağı orta {num} dəyərinə malikdir ({group_means[least]:.2f})."
    ]
    return fig, insight




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


def get_available_graphs(categorical_cols, numeric_cols, datetime_cols, ):
    available_graphs = []

    # Categorical bar charts
    for col in categorical_cols:
        available_graphs.append(f"Bar Chart: {col}")

    # Numerical distributions
    for col in numeric_cols:
        available_graphs.append(f"Histogram: {col}")

    # Time series (only if datetime cols exist)
    if datetime_cols and numeric_cols:
        for dt in datetime_cols:
            for num in numeric_cols:
                available_graphs.append(f"Time Series: {dt} vs {num}")

    # Categorical vs categorical combinations
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            available_graphs.append(f"Stacked Bar: {categorical_cols[i]} vs {categorical_cols[j]}")

    # Categorical vs categorical combinations
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            available_graphs.append(f"Heat Map: {categorical_cols[i]} vs {categorical_cols[j]}")

    # Numeric vs Numeric (scatter)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            available_graphs.append(f"Scatter: {numeric_cols[i]} vs {numeric_cols[j]}")

    # Numeric vs Categorical (box plots)
    for num in numeric_cols:
        for cat in categorical_cols:
            available_graphs.append(f"Box Plot: {num} vs {cat}")

        # Numeric vs Categorical (violin plots)
    for num in numeric_cols:
        for cat in categorical_cols:
            available_graphs.append(f"Violin Plot: {num} vs {cat}")

        # Numeric vs Categorical (bar plots)
    for num in numeric_cols:
        for cat in categorical_cols:
            available_graphs.append(f"Bar Plot: {num} vs {cat}")

    return available_graphs


def generate_graph(df, desc):
    if desc.startswith("Bar Chart: "):
        col = desc.replace("Bar Chart: ", "")
        return plot_bar_chart(df, col)

    elif desc.startswith("Histogram: "):
        col = desc.replace("Histogram: ", "")
        return plot_histogram(df, col)

    elif desc.startswith("Time Series: "):
        dt, num = desc.replace("Time Series: ", "").split(" vs ")
        return plot_time_series(df, dt, num)

    elif desc.startswith("Stacked Bar: "):
        cat1, cat2 = desc.replace("Stacked Bar: ", "").split(" vs ")
        return plot_stacked_bar(df, cat1, cat2)

    elif desc.startswith("Heat Map: "):
        cat1, cat2 = desc.replace("Heat Map: ", "").split(" vs ")
        return plot_heat_map(df, cat1, cat2)

    elif desc.startswith("Scatter: "):
        x, y = desc.replace("Scatter: ", "").split(" vs ")
        return plot_scatter(df, x, y)

    elif desc.startswith("Box Plot: "):
        num, cat = desc.replace("Box Plot: ", "").split(" vs ")
        return plot_boxplot(df, num, cat)

    elif desc.startswith("Violin Plot: "):
        num, cat = desc.replace("Violin Plot: ", "").split(" vs ")
        return plot_violinplot(df, num, cat)

    elif desc.startswith("Bar Plot: "):
        num, cat = desc.replace("Bar Plot: ", "").split(" vs ")
        return plot_barplot_num_vs_cat(df, num, cat)

    return None, ["Plot not supported."]

from PIL import Image

def create_pdf(plot_files, report_title, insights=None, output_file="gpt_data_report_new.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)


    font_path = download_dejavu_font()
    pdf.add_font("FreeSans", "", font_path, uni=True)

    # First page: Title and Date
    pdf.add_page()
    pdf.set_font("FreeSans", '', 24)
    pdf.cell(0, 40, report_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    months_az = {
        "January": "Yanvar", "February": "Fevral", "March": "Mart", "April": "Aprel",
        "May": "May", "June": "İyun", "July": "İyul", "August": "Avqust",
        "September": "Sentyabr", "October": "Oktyabr", "November": "Noyabr", "December": "Dekabr"
    }
    today = datetime.today()
    month_az = months_az[today.strftime("%B")]
    date_str = f"Tarix: {today.day} {month_az} {today.year}"

    pdf.set_font("FreeSans", '', 14)
    pdf.cell(0, 10, date_str, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')


    if insights is None:
        insights = [""] * len(plot_files)


    num_pages = min(len(plot_files), len(insights))

    for i in range(num_pages):
        pdf.add_page()
        img_path = plot_files[i]
        # Calculate scaled height based on image dimensions
        desired_width = pdf.w - 30
        with Image.open(img_path) as im:
            orig_width, orig_height = im.size
        scaled_height = (orig_height / orig_width) * desired_width

        # Add some space before image
        pdf.ln(10)

        # Insert image at current y, and record Y position
        current_y = pdf.get_y()
        pdf.image(img_path, x=15, y=current_y, w=desired_width)

        # Move cursor below the image
        pdf.set_y(current_y + scaled_height + 10)

        # Key Insights - bold header + bullet points
        pdf.set_font("FreeSans", '', 14)
        pdf.cell(0, 10, "Əsas Məqamlar:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("FreeSans", '', 12)
        # If insights[i] is a list, print bullets
        page_width = pdf.w - 2 * pdf.l_margin  # usable width within margins

        if isinstance(insights[i], (list, tuple)):
            for insight in insights[i]:
                x_start = pdf.get_x()
                pdf.multi_cell(page_width, 8, f"- {insight}")
                pdf.set_x(x_start)  # reset x so it doesn’t shift right
        else:
            pdf.multi_cell(page_width, 8, f"- {insights[i]}")
        pdf.ln(3)


    for page_num in range(1, num_pages + 1):
        pdf.page = page_num
        pdf.set_y(-15)
        pdf.set_font("FreeSans", '', 10)
        pdf.cell(0, 10, f"Səhifə {page_num} / {num_pages}", align='C')

    pdf.output(output_file)
    print(f"Report saved as: {output_file}")
    return output_file

def get_graphs_and_insights(selected_graphs, df):
    all_insights = []
    image_paths = []
    for i in range(len(selected_graphs)):
        g = selected_graphs[i]
        data = generate_graph(df, g)
        fig = data[0]
        insight = data[1]

        filename = f"chart_{i}.png"
        fig.savefig(filename)
        plt.close(fig)

        all_insights.append(insight)
        image_paths.append(filename)

    return all_insights, image_paths





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
                    f"Filter {col} by date range", value=[min_date, max_date]
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    if start_date == end_date:
                        st.warning("Please select a different end date to create a valid range.")
                    else:
                        filters[col] = (start_date, end_date)
                else:
                    st.warning("Please select both start and end dates.")
            elif pd.api.types.is_numeric_dtype(dtype):
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                val_range = st.slider(f"Filter {col} range", min_val, max_val, (min_val, max_val))
                filters[col] = val_range
            elif pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                options = df[col].dropna().unique().tolist()
                if df[col].isna().any():
                    options.append("(Missing)")
                selected = st.multiselect(f"Filter {col} categories", options, default=options)
                filters[col] = selected


        # Apply filters
        filtered_df = df.copy()

        for col, filter_val in filters.items():
            if pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                filtered_df = filtered_df[
                    (filtered_df[col].isna()) |
                    ((filtered_df[col] >= pd.to_datetime(filter_val[0])) & (
                                filtered_df[col] <= pd.to_datetime(filter_val[1])))
                    ]
            elif pd.api.types.is_numeric_dtype(df[col].dtype):
                filtered_df = filtered_df[
                    (filtered_df[col].isna()) |
                    ((filtered_df[col] >= filter_val[0]) & (filtered_df[col] <= filter_val[1]))
                    ]
            else:
                if "(Missing)" in filter_val:
                    selected_vals = [val for val in filter_val if val != "(Missing)"]
                    filtered_df = filtered_df[
                        filtered_df[col].isin(selected_vals) | filtered_df[col].isna()
                        ]
                else:
                    filtered_df = filtered_df[filtered_df[col].isin(filter_val)]

        st.write("### Filtered Data")
        st.dataframe(filtered_df)

        st.write("### Cleaned Data")
        cleaned_df = filtered_df.copy()
        cleaned_df = clean_data(cleaned_df)
        st.dataframe(cleaned_df)


        st.write("### Graphs for Filtered and Cleaned Data")

        if not cleaned_df.empty:
            # Example 1: Histogram for numeric column
            numeric_cols = cleaned_df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                st.subheader("Rəqəmsal Analiz - Histogram")
                selected_col = st.selectbox("Histogram üçün ədədi sütun seçin", numeric_cols)
                fig1, insight1 = plot_histogram(cleaned_df, selected_col)
                all_figures.append(fig1)
                st.pyplot(fig1)
                all_insights.append(insight1)
                for i in insight1:
                    st.info(i)

            # Example 2: Time Series Plot for datetime column
            datetime_cols = cleaned_df.select_dtypes(include='datetime').columns.tolist()
            if datetime_cols and numeric_cols:
                st.subheader("Zaman seriyası qrafiki")
                dt_col = st.selectbox("Tarix/zaman sütunu", datetime_cols)
                num_col = st.selectbox("Ədədi sütun", numeric_cols, key="ts_plot")
                fig2, insight2 = plot_time_series(cleaned_df, dt_col, num_col)
                all_figures.append(fig2)
                st.pyplot(fig2)
                all_insights.append(insight2)
                for i in insight2:
                    st.info(i)

            # Example 3: Bar Plot for category counts
            cat_cols = cleaned_df.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                st.subheader("Kateqorial Analiz - Bar Plot")
                cat_col = st.selectbox("Sütun üzrə sayların bar qrafiki", cat_cols)
                fig3, insight3 = plot_bar_chart(cleaned_df, cat_col)
                all_figures.append(fig3)
                st.pyplot(fig3)
                all_insights.append(insight3)
                for i in insight3:
                    st.info(i)


            if cat_cols and numeric_cols:
                st.subheader("Rəqəmsal vs Kateqorial Analiz")

                x_cat = st.selectbox("Kateqorial sütun seçin", cat_cols)
                y_num = st.selectbox("Ədədi sütun seçin", numeric_cols)

                plot_type = st.radio("Qrafik növünü seçin:", ["Box Plot", "Violin Plot", "Bar Plot"], horizontal=True)

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
                st.subheader("Kateqorial vs Kateqorial Analiz")
                x_cat = st.selectbox("Kateqorial sütun seçin (X oxu)", cat_cols, key="cat_vs_cat_x")
                y_cat = st.selectbox("Kateqorial sütun seçin (Y üçün)", [col for col in cat_cols if col != x_cat],
                                     key="cat_vs_cat_y")
                if x_cat and y_cat:

                    chart_type = st.radio("Qrafik növünü seçin:", ["Stacked Bar", "Heatmap"], horizontal=True)

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
                st.subheader("Rəqəmsal vs Rəqəmsal Analiz")
                x_num = st.selectbox("Rəqəmsal sütun seçin (X oxu)", numeric_cols, key="num_vs_num_x")
                y_num = st.selectbox("Rəqəmsal sütun seçin (Y oxu)", [col for col in numeric_cols if col != x_num],
                                     key="num_vs_num_y")

                if x_num and y_num:
                    fig6, insight6, ax = plot_scatter(cleaned_df, x_num, y_num)


                    add_reg = st.checkbox("Regressiya xətti əlavə et", value=False)
                    if add_reg:
                        sns.regplot(data=cleaned_df, x=x_num, y=y_num, scatter=False, ax=ax, color='red')

                    st.pyplot(fig6)

                    all_figures.append(fig6)
                    all_insights.append(insight6)
                    for i in insight6:
                        st.info(i)

        st.markdown("---")
        st.subheader("Hesabat Yarat")

        available_graphs = get_available_graphs(cat_cols, numeric_cols, datetime_cols)

        selected_graphs = st.multiselect(
            "PDF-ə daxil etmək istədiyiniz spesifik qrafik nümunələrini seçin:",
            options=available_graphs
        )

        if selected_graphs:
            report_title = st.text_input("Hesabat üçün başlıq daxil edin:", value="Data Analiz Hesabatı")

            if st.button("Hesabat PDF Yarat"):

                all_insights, all_graphs = get_graphs_and_insights(selected_graphs, cleaned_df)
                pdf_data = create_pdf(all_graphs, report_title, all_insights)
                with open(pdf_data, "rb") as f:
                    pdf_bytes = f.read()
                st.success("Hesabat yaradıldı!")

                st.download_button(
                    label="PDF Hesabatı Yüklə",
                    data=pdf_bytes,
                    file_name="selected_graphs_report.pdf",
                    mime="application/pdf"
                )

        st.markdown("---")
        st.subheader("Bir neçə sual ver (GPT tərəfindən cavablandırılacaq)")
        if "question_count" not in st.session_state:
            st.session_state.question_count = 1

        # Button to add a new question section
        if st.button("+ Yeni Sual"):
            st.session_state.question_count += 1

        # Collect all questions from user
        questions = []
        for i in range(st.session_state.question_count):
            question = st.text_input(f"Sual {i + 1}:", key=f"q_{i}")
            if question.strip():
                questions.append(question)


        if st.button("Cavab Al"):
            if questions:
                combined_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
                with st.spinner("GPT cavabı hazırlanır..."):
                    gpt_answer = get_response(combined_questions, cleaned_df)
                    answers = parse_gpt_response(gpt_answer, cleaned_df)
                    st.session_state["last_answers"] = answers


            else:
                st.warning("Zəhmət olmasa bir sual daxil edin.")

        if "last_answers" in st.session_state:
            st.subheader("Cavablar:")
            for question, answer in st.session_state["last_answers"].items():
                st.markdown(f"**Sual:** {question}")
                st.markdown(f"**Cavab:** {answer}")
                st.markdown("---")

        st.write(cleaned_df.columns)
        st.write(cleaned_df.isna().sum())
        st.write(cleaned_df.nunique(dropna=False))




if __name__ == "__main__":
    main()