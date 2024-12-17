import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def generate_response(query, uploaded_files):

    load_dotenv()  # takes variables from .env file

    # 1. Loaded PDF
    print("1. Loading PDF files...")
    # file_path = "databricks-llm.pdf"
    print(uploaded_files)

    all_docs = []

    # accepts multiple files
    for file in uploaded_files:
        loader = PyPDFLoader(f"files/{file.name}")
        documents = loader.load()

        # 2. Split document into chunks - for precise matching and fitting context window
        print("\n2. Chunking documents... ")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents=documents)

        all_docs.extend(split_documents)

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
        search_kwargs={'k': 6, 'lambda_mult': 0.25, 'fetch_k': 20}
    )

    # query = "How are companies using LLMs?"
    print(f"Question: {query}\n")

    # retrieve the most relevant documents
    docs = retriever.invoke(query)

    context = '\n\n'.join([doc.page_content for doc in docs])
    print(context)

    # 6. Set up LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.01,
        max_tokens=512,
        max_retries=2,
    )

    # 7. Create prompt
    template = (
        "Answer the question based only on the following context below."
        "Please say you don't know if you cannot find the information from the context.\n"
        "If you see many numbers in a sentence, they are actually a table row with "
        "space-separated values. The first word is usually the column name and the "
        "numbers on the right are the values.\n"
        "The header row of this table may be 1 or 2 sentences above.\n"
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
    return result.content
