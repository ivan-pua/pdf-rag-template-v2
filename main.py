import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def generate_response(query, uploaded_files):
    load_dotenv()  # takes variables from .env file

    # Clear existing vector store before processing new PDF
    vector_store_folder_path = "faiss_index"
    if os.path.exists(vector_store_folder_path):
        shutil.rmtree(vector_store_folder_path)

    # Ensure files directory exists
    os.makedirs("files", exist_ok=True)

    # Explicitly save uploaded file
    file_path = f"files/{uploaded_files.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_files.getvalue())

    # 1. Load PDF
    print("1. Loading PDF files...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(len(documents))
    print(documents[0].page_content[0:100])
    print(documents[0].metadata)

    # 2. Split document into chunks
    print("\n2. Chunking documents... ")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents=documents)
    print(len(split_documents))
    print(split_documents[0])

    openai_key = os.getenv("OPENAI_API_KEY")

    # 3. Create embeddings
    print("\n3. Create embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. Create new vector store
    print("\n4. Create new vector store... ")
    vectordb = FAISS.from_documents(split_documents, embeddings)
    vectordb.save_local(vector_store_folder_path)

    # 5. Retriever
    print("\n5. Creating a retriever... ")
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'lambda_mult': 0.25}
    )

    print(f"Question: {query}\n")

    # retrieve the most relevant documents
    docs = retriever.invoke(query)

    context = '\n\n'.join([doc.page_content for doc in docs])
    print(context)

    # 6. Set up LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=512,
        max_retries=2,
    )

    # 7. Create prompt
    template = """Answer the question based only on the following context below. Please say you don't know if you cannot find the information from the context.
    <context>
    {context}
    </context>"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    result = chain.invoke(
        {
            "context": context,
            "input": query,
        }
    )

    print("Answer: ")
    print(result.content)
    return result.content
