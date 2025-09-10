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
     # Buttons for adding/clearing questions
     col1, col2 = st.columns([1, 1])
        
     with col1:
          if st.button("+ Add Question"):
               st.session_state.question_count += 1
        
     with col2:
          if st.button("Clear All"):
               # Reset to 1 question and clear all stored values
               old_count = st.session_state.question_count
               st.session_state.question_count = 1
               # Clear all question values from session state
               for i in range(old_count):
                    if f"q_{i}" in st.session_state:
                        del st.session_state[f"q_{i}"]
               st.rerun()  # Refresh to show cleared inputs
   
        
     # Collect questions from user
     questions = []
     for i in range(st.session_state.question_count):
          # Create columns for question input and delete button
          if st.session_state.question_count > 1 and i > 0:
               col_q, col_del = st.columns([4, 1])
               with col_q:
                    question = st.text_input(f"Question {i + 1}:", key=f"q_{i}")
               with col_del:
                    st.write("")  # Add some spacing
                    if st.button("Delete", key=f"del_{i}", help="Delete this question"):
                        delete_question(i)
                        st.rerun()
            else:
                # First question cannot be deleted
                question = st.text_input(f"Question {i + 1}:", key=f"q_{i}")
            
            if question.strip():
                questions.append(question)
        
  
        
     return questions

def clear_questions_after_answer():
     # Clear all question values
     for i in range(st.session_state.question_count):
          if f"q_{i}" in st.session_state:
               del st.session_state[f"q_{i}"]
        
     # Reset to just one empty question field
     st.session_state.question_count = 1
     st.rerun()  # Refresh the interface

def delete_question(index_to_delete):
        
     # Get all current questions
     current_questions = []
     for i in range(st.session_state.question_count):
          if f"q_{i}" in st.session_state:
               current_questions.append(st.session_state[f"q_{i}"])
          else:
               current_questions.append("")
        
     # Remove the question at the specified index
     if 0 <= index_to_delete < len(current_questions):
          current_questions.pop(index_to_delete)
        
     # Clear all existing question keys
     for i in range(st.session_state.question_count):
          if f"q_{i}" in st.session_state:
               del st.session_state[f"q_{i}"]
        
     # Reassign remaining questions to new keys
     st.session_state.question_count = max(1, len(current_questions))  # At least 1
     for i, question in enumerate(current_questions):
          st.session_state[f"q_{i}"] = question

def handle_answer_generation(df):
    questions = show_question_interface()
        
    if st.button("Get Answer"):
        if questions:
            generate_answers(questions, df)
            clear_questions_after_answer()
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




