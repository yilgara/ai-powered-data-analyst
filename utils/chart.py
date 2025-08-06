
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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