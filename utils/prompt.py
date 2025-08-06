import google.generativeai as genai
import re
from google.generativeai.types import GenerationConfig
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


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


def get_response(user_questions, df, model_name='gemini-1.5-flash'):
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
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