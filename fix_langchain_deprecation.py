#!/usr/bin/env python3
"""
This script patches the PDF reader code to fix LangChain deprecation warnings
by updating all instances of the deprecated 'run' method to use 'invoke' instead.
"""
import re
import os

def fix_langchain_deprecation():
    file_path = os.path.join('pdf', '__init__.py')
    
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Fix RetrievalQA chain creation
    content = re.sub(
        r'qa_chain = RetrievalQA\.from_chain_type\(llm=llm, retriever=retriever\)',
        'qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)',
        content
    )
    
    # Replace any remaining instances of .run() with .invoke()
    content = re.sub(
        r'(\w+)\.run\(([^)]+)\)',
        r'\1.invoke({"query": \2})',
        content
    )
    
    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print("✅ Successfully updated PDF reader code to fix LangChain deprecation warnings")

if __name__ == "__main__":
    fix_langchain_deprecation()
