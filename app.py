import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import GooglePalmEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from flask import Flask, request, send_file, render_template
from tempfile import NamedTemporaryFile

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Configure palm with the provided API key
import google.generativeai as palm
palm.configure(api_key=os.environ.get('API_KEY'))

# Initialize GooglePalm Language Model
from langchain.llms import GooglePalm
llm = GooglePalm()
llm.temperature = 0.1
llm.max_output_tokens = 1024

# Flask app setup
app = Flask(__name__)

# Function to generate the prompt for extracting clauses
def get_clauses_prompt(txt):
    prompt = f"""Extract clauses present with little detail in the given text. Also mention the type of agreement.
Given text: {txt}
Output:"""
    return prompt

# Function to generate the prompt for clause retrieval
def retrieval_prompt(question, context):
    prompt = f"""Given piece of document: {question}
If the given piece of agreement is a legal agreement, then: You are a law expert, given an Agreement, you have to find the most important missing clauses, and write content for the missing clauses according to the given agreement. You can have reference documents similar to the given agreement; you can utilize them for an answer.
References: {context}

If the given document is not a legal agreement: proceed as an assistant.

Final Answers:
"""
    return prompt

# Function to split text into chunks and process the clauses
def process_clauses(input_docs_path, output_file_path):
    loader = Docx2txtLoader(input_docs_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    f_prompt = []
    for chunk in chunks:
        prompt = get_clauses_prompt(chunk.page_content)
        f_prompt.append(prompt)

    llm_result = llm._generate(f_prompt)
    final_clauses = ""
    for generation in llm_result.generations:
        if generation:
            final_clauses += "\n" + generation[0].text

    # Split final_clauses into chunks for similarity search
    f_text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=60)
    f_chunks = f_text_splitter.split_text(final_clauses)

    # Load training vectors for similarity search
    embedding_function = GooglePalmEmbeddings()
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    total_prompt = []
    for f_chunk in f_chunks:
        docs = db.similarity_search(f_chunk)
        prompt = retrieval_prompt(f_chunk, docs[:2])
        total_prompt.append(prompt)

    # Get missing clauses using language model
    results = llm._generate(total_prompt)

    missing_clause = ""
    for generation in results.generations:
        if generation:
            missing_clause += "\n" + generation[0].text

    # Get final output of the model
    final_out = llm._generate([f"Content: {missing_clause}\nRewrite the content and remove repetition if present. Do not remove contents; give output with proper document"]).generations[0][0].text

    # Save the final output to a temporary docx file
    with NamedTemporaryFile(suffix='.docx', delete=False) as temp_output:
        temp_output_path = temp_output.name
        text_to_docx(final_out, temp_output_path)
        return temp_output_path

def text_to_docx(text, output_file_path):
    import docx
    # Create a new docx document
    doc = docx.Document()
    # Add the text to the docx document
    doc.add_paragraph(text)
    # Save the docx document
    doc.save(output_file_path)
    print("Saved")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'input_file' not in request.files:
            return render_template('index.html', error='No file part')

        input_file = request.files['input_file']

        # Check if the file is empty
        if input_file.filename == '':
            return render_template('index.html', error='No selected file')

        # Save the input file to a temporary location
        with NamedTemporaryFile(suffix='.docx', delete=False) as temp_input:
            temp_input_path = temp_input.name
            input_file.save(temp_input_path)

        # Process the clauses and get the output file path
        output_file_path = process_clauses(temp_input_path, "output.docx")

        # Remove the temporary input file
        os.remove(temp_input_path)

        # Return the processed output file for download
        return send_file(output_file_path, as_attachment=True, download_name="output.docx")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
