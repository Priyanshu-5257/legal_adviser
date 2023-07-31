import os
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import Docx2txtLoader
import google.generativeai as palm

load_dotenv(find_dotenv())  # read local .env file
palm.configure(api_key=api_key)

llm = palm.GooglePalm()
llm.temperature = 0.15
llm.max_output_tokens = 1024

def get_clauses(txt):
    prompt = f""" Extract clauses present with little detail in the given text. Also, mention the type of agreement.
    Given text: {txt}

    Output:"""
    return prompt

list_docs = os.listdir("SampleDocs/")
list_docs = [l for l in list_docs if l.endswith("docx")]
final_learning = ""

for doc_path in list_docs:
    loader = Docx2txtLoader("SampleDocs/" + doc_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    f_prompt = []

    for chunk in chunks:
        prompt = get_clauses(chunk.page_content)
        f_prompt.append(prompt)

    llm_result = llm._generate(f_prompt)
    current_learning = ""

    for gen in llm_result.generations:
        if len(gen) != 0:
            current_learning += "\n" + gen[0].text
        else:
            print("Output is None.")

    output_file_path = "SampleDocs/temp/" + doc_path[:-5] + ".txt"
    with open(output_file_path, 'w') as file:
        file.write(current_learning)
        print(f"Generated texts successfully saved to '{output_file_path}'.")

    final_learning += "\n" + current_learning

output_file_path = "SampleDocs/temp/" + "total_learning.txt"
with open(output_file_path, 'w') as file:
    file.write(final_learning)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_text(final_learning)

# Create the open-source embedding function
embedding_function = GooglePalmEmbeddings()
# Load it into Chroma
db = Chroma.from_texts(docs, embedding_function, persist_directory="./chroma_db")

print("Saved.")
