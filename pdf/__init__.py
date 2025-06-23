import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os

def render_pdf_reader():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Lato:wght@400;700&display=swap');
        """ + open("pdf/style.css").read() + """
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Document Analysis")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", key="pdf_api_key")
    st.write("""
    ### Instructions
    - Upload the PDFs and copy/paste website URL's that you would like to analyze
    - A summary will be generated for each source, then you can ask questions
    """)
    uploaded_files = st.file_uploader("Upload PDF document(s)", type="pdf", accept_multiple_files=True, key="pdf_upload")
    website_urls = st.text_area("Enter website URLs (one per line)", key="pdf_urls")

    all_docs = []
    doc_summaries = []
    doc_names = []
    doc_qa_chains = []
    llm = None

    if openai_api_key:
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Handle PDFs
        if uploaded_files:
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    chunks = text_splitter.split_documents(docs)
                    all_docs.append(chunks)
                    doc_names.append(file.name)
                    summary_text = "\n".join([chunk.page_content for chunk in chunks[:5]])
                    summary_prompt = f"Summarize the following document for a customer support context:\n\n{summary_text}"
                    summary = llm.invoke(summary_prompt)
                    doc_summaries.append(summary.content if hasattr(summary, 'content') else summary)
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    retriever = vectorstore.as_retriever()
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    doc_qa_chains.append(qa_chain)
                finally:
                    os.remove(tmp_path)

        # Handle Websites
        url_list = [url.strip() for url in website_urls.splitlines() if url.strip()]
        if url_list:
            for url in url_list:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    chunks = text_splitter.split_documents(docs)
                    all_docs.append(chunks)
                    doc_names.append(url)
                    summary_text = "\n".join([chunk.page_content for chunk in chunks[:5]])
                    summary_prompt = f"Summarize the following website for a customer support context:\n\n{summary_text}"
                    summary = llm.invoke(summary_prompt)
                    doc_summaries.append(summary.content if hasattr(summary, 'content') else summary)
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    retriever = vectorstore.as_retriever()
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    doc_qa_chains.append(qa_chain)
                except Exception as e:
                    st.error(f"Failed to load website {url}: {e}")

    # Show summaries
    if doc_names:
        st.header("Document Summaries")
        for i, name in enumerate(doc_names):
            st.subheader(name)
            st.info(doc_summaries[i])

    # QA for individual docs
    if doc_names:
        st.header("Ask Questions About a Document")
        selected_doc = st.selectbox("Select a document to query:", doc_names, key="pdf_select_doc")
        user_question = st.text_input("Ask a question about the selected document:", key="pdf_single_doc_qa")
        if user_question and openai_api_key:
            idx = doc_names.index(selected_doc)
            response = doc_qa_chains[idx].run(user_question)
            st.write(f"**Answer from {selected_doc}:**", response)

    # Compare/contrast
    if len(doc_names) > 1:
        st.header("Compare/Contrast Documents")
        selected_docs = st.multiselect("Select documents to compare:", doc_names, key="pdf_compare_docs")
        compare_question = st.text_input("Ask a question about the selected documents:", key="pdf_compare_qa")
        if compare_question and selected_docs and openai_api_key:
            # Concatenate chunks from selected docs
            combined_chunks = []
            for name in selected_docs:
                idx = doc_names.index(name)
                combined_chunks.extend(all_docs[idx])
            # New vectorstore for combined docs
            combined_vectorstore = FAISS.from_documents(combined_chunks, embeddings)
            combined_retriever = combined_vectorstore.as_retriever()
            combined_qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=combined_retriever)
            response = combined_qa_chain.run(compare_question)
            st.write("**Answer:**", response)

    # Quiz Generator
    import json
    if doc_names and openai_api_key:
        st.header("Test your Understanding")
        if st.button("Generate Quiz Questions", key="pdf_generate_quiz"):
            with st.spinner("Generating quiz questions using AI..."):
                combined_text = "\n".join(chunk.page_content for doc in all_docs for chunk in doc[:5])
                quiz_prompt = (
                    "Create a quiz of 5 multiple-choice questions based on the following content. "
                    "For each question, provide 4 options (A, B, C, D), indicate the correct answer, and provide a brief explanation for the correct answer. "
                    "Return ONLY valid JSON, no explanation, no markdown, no preamble. "
                    "Format: [ { 'question': '...', 'options': ['A ...', 'B ...', 'C ...', 'D ...'], 'answer': 'A', 'explanation': '...'}, ... ]\n\nContent:"
                    + combined_text
                )
                quiz_response = llm.invoke(quiz_prompt)
                quiz_text = quiz_response.content if hasattr(quiz_response, 'content') else quiz_response
                # Try to extract JSON if wrapped in markdown/code block
                import re
                def extract_json(text):
                    # Remove markdown code block if present
                    match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    # Try to find the first [ ... ] block
                    match = re.search(r'(\[.*\])', text, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    return text.strip()
                quiz_text_clean = extract_json(quiz_text)
                try:
                    quiz_data = json.loads(quiz_text_clean)
                    st.session_state['quiz_data'] = quiz_data
                    st.session_state['quiz_submitted'] = False
                except Exception:
                    st.error("Could not parse quiz questions. Here is the raw output. Please try again:")
                    st.markdown(f"<pre>{quiz_text}</pre>", unsafe_allow_html=True)
                    st.session_state['quiz_data'] = None
                    st.session_state['quiz_submitted'] = False
        quiz_data = st.session_state.get('quiz_data')
        quiz_submitted = st.session_state.get('quiz_submitted', False)
        user_answers = st.session_state.get('quiz_user_answers', [])
        if quiz_data and not quiz_submitted:
            with st.form("quiz_form"):
                user_answers = []
                for idx, q in enumerate(quiz_data):
                    st.write(f"**Q{idx+1}: {q['question']}**")
                    options = q['options']
                    user_choice = st.radio("Select an answer:", options, key=f"quiz_{idx}")
                    user_answers.append(user_choice)
                submitted = st.form_submit_button("Submit Quiz")
                if submitted:
                    st.session_state['quiz_user_answers'] = user_answers
                    st.session_state['quiz_submitted'] = True
                    st.experimental_rerun()

        if quiz_data and quiz_submitted and user_answers:
            score = 0
            for idx, q in enumerate(quiz_data):
                correct_letter = q.get('answer')
                correct_option = [opt for opt in q['options'] if opt.startswith(correct_letter)][0] if correct_letter else None
                explanation = q.get('explanation', 'No explanation provided.')
                user_option = user_answers[idx]
                if user_option.startswith(correct_letter):
                    st.success(f"Q{idx+1}: Correct!")
                    score += 1
                else:
                    st.error(f"Q{idx+1}: Incorrect. Correct answer: {correct_option}")
                st.info(f"Explanation: {explanation}")
            st.info(f"Your score: {score} out of {len(quiz_data)}")
