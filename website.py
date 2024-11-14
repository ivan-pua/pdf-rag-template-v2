import streamlit as st
from main import generate_response

st.set_page_config(page_title="PDF-Bot", page_icon="📖")

st.title("📖 PDF-Bot")
st.caption("🚀 Chat easily with your PDF documents")

# Upload PDF with a unique key to force re-upload
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", 
    type=["pdf"], 
    accept_multiple_files=False,
    key="pdf_uploader"
)

# Query text
query_text = st.text_input(
    'Enter your question:', 
    placeholder='What is the history of LLMs?', 
    disabled=not uploaded_files
)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button(
        'Submit', disabled=not (uploaded_files and query_text)
    )
    
    if submitted:
        with st.spinner('Generating Answer...'):
            try:
                response = generate_response(query_text, uploaded_files)
                result.append(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if len(result):
    st.info(response)
