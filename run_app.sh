#!/bin/bash
# Set the OpenMP environment variable to avoid conflicts
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the fix script to update LangChain deprecated methods
python fix_langchain_deprecation.py

# Run the Streamlit app
streamlit run app.py
