import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="Document Analysis")

# Inject custom CSS for fun but professional color scheme and fonts
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

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

st.write("""
### Instructions
- Upload the PDFs and copy/paste website URL's that you would like to analyze
- A summary will be generated for each source, then you can ask questions
- You can also generate a quiz based on the uploaded material to test your understanding
""")

uploaded_files = st.file_uploader("Upload PDF document(s)", type="pdf", accept_multiple_files=True)
website_urls = st.text_area("Enter website URLs (one per line)")

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
    selected_doc = st.selectbox("Select a document to query:", doc_names)
    user_question = st.text_input("Ask a question about the selected document:", key="single_doc_qa")
    if user_question and openai_api_key:
        idx = doc_names.index(selected_doc)
        response = doc_qa_chains[idx].run(user_question)
        st.write(f"**Answer from {selected_doc}:**", response)

# Compare/contrast
if len(doc_names) > 1:
    st.header("Compare/Contrast Documents")
    selected_docs = st.multiselect("Select documents to compare:", doc_names)
    compare_question = st.text_input("Ask a question about the selected documents:", key="compare_qa")
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
        doc_list_str = ", ".join(selected_docs)
        system_instruction = (
            "You are an expert at comparing and constrasting documents. "
            f"The user has selected the following documents to compare: {doc_list_str}\n\n"
            "Please answer the user's question to the best of your ability.\n"
            "If you cannot answer, please politely tell the user that you are confused by their intention and ask them to rephrase the question.\n"
        )
        full_prompt = system_instruction + compare_question
        response = combined_qa_chain.run(full_prompt)
        st.write(f"**Compare/Contrast Answer:**", response)