import os
from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

AZURE_AI_SEARCH_ENDPONT = os.getenv("AZURE_AI_SEARCH_ENDPONT")
AZURE_AI_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_KEY")
AZURE_AI_SEARCH_INDEX_NAME = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

########################################
# Configure vector DB
########################################

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDING_MODEL,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

vector_store = AzureSearch(
    azure_search_endpoint=AZURE_AI_SEARCH_ENDPONT,
    azure_search_key=AZURE_AI_SEARCH_KEY,
    index_name=AZURE_AI_SEARCH_INDEX_NAME,
    embedding_function=embeddings_model.embed_query,
)

retriever = vector_store.as_retriever(search_type = "similarity")

########################################
# Configure chatbot 
########################################

from langchain.globals import set_verbose, set_debug
#set_debug(True)
set_verbose(True)

from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a chatbot expert in Avaya Experience Platform. You can use the following context from the AXP documentation to answer the user question.

        {context}

        Question: {question}

        Helpful Answer:"""
    )
])

print("Chatbot configured, using prompt template:")
prompt.pretty_print()

def format_docs(docs):
    concatenated_text = "\n\n".join(doc.page_content for doc in docs)
    print(f"Context injected: {concatenated_text}")
    return concatenated_text

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    # creates a dictionary where context value is filled up by retriever then formatted by format_docs
    # and question is passed over unchanged by RunnablePassthrough
    # these are Runnable objects that will be executed in parallel or sequence and the output is fed forward
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #prompt expects dictionary of context and question
    | prompt
    | llm
    | StrOutputParser()
)

########################################
# Configure FastAPI 
########################################

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from typing import Annotated

app = FastAPI(root_path="/", title="Avaya AXP RAG Agent", version="1.0.0")

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/bot")
async def root(question: str, api_key: Annotated[str, Header()]):
    print(f"Received question: {question}")
    print(api_key)
    if api_key != "avaya-hungary":
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    response =  rag_chain.invoke(question)
    return {"response": response}
