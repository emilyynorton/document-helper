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
    - You can also compare/contrast documents and generate a quiz
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
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
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
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
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
            # Use invoke method instead of deprecated run method
            response = doc_qa_chains[idx].invoke({"query": user_question})
            response_text = response.get('result', response) if isinstance(response, dict) else response
            st.write(f"**Answer from {selected_doc}:**", response_text)

    # Compare/contrast
    if len(doc_names) > 1:
        st.header("Compare/Contrast Documents")
        selected_docs = st.multiselect("Select documents to compare:", doc_names)
        compare_question = st.text_input("Ask a question about the selected documents:", key="compare_qa")
        if compare_question and len(selected_docs) >= 2 and openai_api_key:
            with st.spinner("Analyzing and comparing documents..."):
                # Extract content from selected documents
                doc_contents = {}
                for name in selected_docs:
                    idx = doc_names.index(name)
                    doc_chunks = all_docs[idx]
                    # Get the first few chunks to represent the document
                    doc_content = "\n\n".join([chunk.page_content for chunk in doc_chunks[:3]])
                    doc_contents[name] = doc_content
                
                # Create a prompt that explicitly includes document content
                comparison_prompt = f"""You are an expert at analyzing and comparing documents. 
                You have been provided with content from {len(selected_docs)} different documents.
                
                Here are the documents:
                """
                
                # Add each document's content to the prompt
                for doc_name, content in doc_contents.items():
                    comparison_prompt += f"\n\nDOCUMENT: {doc_name}\n{content}\n"
                
                # Add the question
                comparison_prompt += f"\n\nQUESTION: {compare_question}\n\n"
                comparison_prompt += """Please provide a detailed comparison of these documents in relation to the question. 
                Include similarities and differences between the documents.
                If the documents don't contain relevant information to answer the question, please state that clearly."""
                
                # Use the LLM directly
                response = llm.invoke(comparison_prompt)
                response_text = response.content if hasattr(response, 'content') else response
                
                st.write(f"**Compare/Contrast Analysis:**", response_text)

    # Quiz Generator
    import json
    if doc_names and openai_api_key:
        st.header("Test your Understanding")
        
        # Always select a specific document for the quiz
        selected_quiz_doc = st.selectbox("Select document for quiz:", doc_names, key="selected_quiz_doc")
        
        quiz_difficulty = st.select_slider(
            "Quiz difficulty:",
            options=["Easy", "Medium", "Hard"],
            value="Medium",
            key="quiz_difficulty"
        )
        
        num_questions = st.slider("Number of questions:", min_value=3, max_value=10, value=5, step=1, key="num_questions")
        
        if st.button("Generate Quiz Questions", key="pdf_generate_quiz"):
            with st.spinner(f"Generating {quiz_difficulty.lower()} quiz with {num_questions} questions about {selected_quiz_doc}..."):
                # Get text from the selected document
                doc_idx = doc_names.index(selected_quiz_doc)
                quiz_text = "\n".join(chunk.page_content for chunk in all_docs[doc_idx][:10])
                quiz_source = selected_quiz_doc
                
                st.session_state['quiz_source'] = quiz_source
                
                quiz_prompt = (
                    f"Create a quiz of {num_questions} {quiz_difficulty.lower()}-difficulty multiple-choice questions based on the following content from {quiz_source}. "
                    f"For {quiz_difficulty.lower()} difficulty: "
                    f"{'Focus on basic facts and definitions' if quiz_difficulty == 'Easy' else 'Include some analytical questions that require understanding concepts' if quiz_difficulty == 'Medium' else 'Include challenging questions that require deep understanding and analysis'}. "
                    "For each question, provide 4 options (A, B, C, D). IMPORTANT: One of these options MUST be the correct answer. "
                    "For each question, indicate the correct answer letter (A, B, C, or D), and provide a brief explanation for why it's correct. "
                    "IMPORTANT: Make absolutely sure that the correct answer letter you specify (A, B, C, or D) corresponds to one of the options you provided. "
                    "Return ONLY valid JSON, no explanation, no markdown, no preamble. "
                    "Format: [ { 'question': '...', 'options': ['A. Option text', 'B. Option text', 'C. Option text', 'D. Option text'], 'answer': 'A', 'explanation': '...'}, ... ]\n\nContent:"
                    + quiz_text
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
                    # Try to fix common JSON issues
                    text = text.replace("'", '"')  # Replace single quotes with double quotes
                    text = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', text)  # Add quotes to keys
                    return text.strip()
                
                quiz_text_clean = extract_json(quiz_text)
                
                try:
                    quiz_data = json.loads(quiz_text_clean)
                    
                    # Validate quiz data structure
                    valid_quiz = True
                    for i, q in enumerate(quiz_data):
                        # Check if question has all required fields
                        if not isinstance(q, dict) or 'question' not in q or 'options' not in q or 'answer' not in q:
                            st.warning(f"Question {i+1} is missing required fields. Fixing format...")
                            valid_quiz = False
                            # Try to fix the question format
                            if 'options' not in q or not isinstance(q['options'], list):
                                q['options'] = [f"A. No option A", f"B. No option B", f"C. No option C", f"D. No option D"]
                            if 'answer' not in q or not isinstance(q['answer'], str):
                                q['answer'] = "A"
                            if 'explanation' not in q:
                                q['explanation'] = "No explanation provided."
                        else:
                            # Ensure options are properly formatted with A, B, C, D prefixes
                            fixed_options = []
                            for j, opt in enumerate(q['options']):
                                prefix = chr(65 + j)  # A, B, C, D...
                                if not opt.startswith(f"{prefix}.") and not opt.startswith(f"{prefix} "):
                                    fixed_options.append(f"{prefix}. {opt.lstrip(f'{prefix}').lstrip('. ')}") 
                                else:
                                    fixed_options.append(opt)
                            q['options'] = fixed_options[:4]  # Ensure exactly 4 options
                            
                            # Ensure answer is just the letter
                            if len(q['answer']) > 1:
                                q['answer'] = q['answer'][0].upper()
                            
                            # Verify that the answer letter corresponds to an option
                            answer_letter = q['answer'].upper()
                            if not any(opt.startswith(answer_letter) for opt in q['options']):
                                st.warning(f"Question {i+1}: Answer letter '{answer_letter}' doesn't match any option. Setting to first option.")
                                q['answer'] = 'A'
                                valid_quiz = False
                    
                    st.session_state['quiz_data'] = quiz_data
                    st.session_state['quiz_submitted'] = False
                    
                    if valid_quiz:
                        st.success("Quiz generated successfully! Answer the questions below.")
                    else:
                        st.warning("Quiz generated with some formatting issues. We've fixed them, but answers may not be accurate.")
                        
                except Exception as e:
                    st.error(f"Could not parse quiz questions: {str(e)}. Here is the raw output. Please try again:")
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
                    st.rerun()

        if quiz_data and quiz_submitted and user_answers:
            quiz_source = st.session_state.get('quiz_source', 'documents')
            st.subheader(f"Quiz Results (Source: {quiz_source})")
            score = 0
            for idx, q in enumerate(quiz_data):
                correct_letter = q.get('answer')
                # Find the correct option with better error handling
                try:
                    matching_options = [opt for opt in q.get('options', []) if opt.startswith(correct_letter)] if correct_letter else []
                    correct_option = matching_options[0] if matching_options else f"{correct_letter} (option text not found)"
                except Exception as e:
                    correct_option = f"{correct_letter} (error: {str(e)})"
                
                explanation = q.get('explanation', 'No explanation provided.')
                user_option = user_answers[idx] if idx < len(user_answers) else "No answer"
                
                if correct_letter and user_option.startswith(correct_letter):
                    st.success(f"Q{idx+1}: Correct!")
                    score += 1
                else:
                    st.error(f"Q{idx+1}: Incorrect. Correct answer: {correct_option}")
                st.info(f"Explanation: {explanation}")
            st.info(f"Your score: {score} out of {len(quiz_data)}")
            if st.button("Generate New Quiz", key="new_quiz_button"):
                st.session_state['quiz_data'] = None
                st.session_state['quiz_submitted'] = False
                st.session_state['quiz_user_answers'] = []
                st.rerun()
