import streamlit as st
from utils.prompt import get_response, parse_gpt_response

def show_qa_section(df):
     
    st.markdown("---")
    st.subheader("Ask a few questions (GPT will answer)")
        
    initialize_session_state()
    handle_answer_generation(df)
    display_previous_answers()


def initialize_session_state():
    if "question_count" not in st.session_state:
        st.session_state.question_count = 1


def show_question_interface():
    # Button to add new question
    if st.button("+ New Question"):
        st.session_state.question_count += 1
        
    # Collect questions from user
    questions = []
    for i in range(st.session_state.question_count):
        question = st.text_input(f"Question {i + 1}:", key=f"q_{i}")
        if question.strip():
            questions.append(question)
        
    return questions


def handle_answer_generation(df):
    questions = show_question_interface()
        
    if st.button("Get Answer"):
        if questions:
            generate_answers(questions, df)
        else:
            st.warning("Please enter a question.")


def generate_answers(questions, df):
    try:
        combined_questions = "\n".join([f"{i + 1}. {q}" for i, q in enumerate(questions)])
            
        with st.spinner("Preparing GPT response..."):
            gpt_answer = get_response(combined_questions, df)
            answers = parse_gpt_response(gpt_answer, df)
            st.session_state["last_answers"] = answers
          
    except Exception as e:
        st.error(f"Error generating answers: {str(e)}")
        st.info("Please try again or check your API configuration.")


def display_previous_answers():
    if "last_answers" in st.session_state:
        st.subheader("Answers:")
        for question, answer in st.session_state["last_answers"].items():
            st.markdown(f"**Question:** {question}")
            st.markdown(f"**Answer:** {answer}")
            st.markdown("---")




