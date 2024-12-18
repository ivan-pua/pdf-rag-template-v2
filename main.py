import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def generate_response(query: str, uploaded_files: list):

    load_dotenv()  # takes variables from .env file

    # 1. Loaded PDF
    print("1. Loading PDF files...")
    # file_path = "databricks-llm.pdf"
    all_docs = []

    # accepts multiple files
    for file in uploaded_files:
        loader = PyPDFLoader(f"files/{file.name}")
        documents = loader.load()

        # 2. Split document into chunks - for precise matching and fitting context window
        print("\n2. Chunking documents... ")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=20)
        split_documents = text_splitter.split_documents(documents=documents)

        all_docs.extend(split_documents)
    
    print(len(all_docs))

    # 3. Create embeddings
    print("\n3. Create embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. Save embeddings in vector store
    vector_store_folder_path = "faiss_index"
    if os.path.isdir(vector_store_folder_path):

        print("\n4. Loading existing vector store... ")
        vectordb = FAISS.load_local(
            vector_store_folder_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:
        print("\n4. Create new vector store... ")
        vectordb = FAISS.from_documents(all_docs, embeddings)
        vectordb.save_local(vector_store_folder_path)

    # 5. Retriever
    print("\n5. Creating a retriever... ")
    # by default uses cosine similarity
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            'k': 5,
            'lambda_mult': 0.75, # the closer to 0, the more diverse
            'fetch_k': 30
        }
    )

    # query = "How are companies using LLMs?"
    print(f"Question: {query}\n")

    # retrieve the most relevant documents
    docs = retriever.invoke(query)

    relevant_docs = [doc.page_content for doc in docs]
    context = '\n\n'.join(relevant_docs)

    # 6. Set up LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.01,
        max_tokens=1024,
        max_retries=2,
    )

    # 7. Create prompt
    template = (
        "Answer the question based only on the following context below."
        "Please say you don't know if you cannot find the information from the context.\n"
        "<context>\n"
        "{context}\n"
        "</context>"
    )

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
    return result.content, relevant_docs
