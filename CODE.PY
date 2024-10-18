# Step 1: Install necessary libraries
!pip install PyPDF2 python-docx pandas transformers

# Step 2: Import required libraries
import PyPDF2
import docx
import pandas as pd
from transformers import pipeline
from google.colab import files
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 3: Functions to extract text from different file formats
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    text = df.to_string(index=False)
    return text

def extract_text_from_file(file_name):
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_name)
    elif file_name.endswith('.docx'):
        return extract_text_from_docx(file_name)
    elif file_name.endswith('.csv'):
        return extract_text_from_csv(file_name)
    else:
        return "Unsupported file format!"

# Step 4: Function to split large text into smaller chunks
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 5: Use TF-IDF to select the most relevant chunks for a question
def get_relevant_chunk(chunks, question):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    query_vec = vectorizer.transform([question])
    
    # Compute cosine similarity between the query and chunks
    cosine_similarities = np.dot(query_vec, tfidf_matrix.T).toarray().flatten()
    most_relevant_idx = cosine_similarities.argmax()
    return chunks[most_relevant_idx]

# Step 6: Load a smaller QA model from Hugging Face
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 7: Function to answer a question based on the most relevant chunk
def answer_question(context, question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Step 8: Upload a file
uploaded_file = files.upload()

# Step 9: Extract text from the uploaded file
file_name = list(uploaded_file.keys())[0]
file_content = extract_text_from_file(file_name)

# Step 10: Chunk the text into smaller parts for faster querying
chunks = chunk_text(file_content)

# Step 11: Ask questions continuously until the user wants to exit
while True:
    question = input("Enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        print("Exiting the QA system.")
        break
    
    # Step 12: Get the most relevant chunk and generate the answer
    relevant_chunk = get_relevant_chunk(chunks, question)
    answer = answer_question(relevant_chunk, question)
    print(f"Answer: {answer}")