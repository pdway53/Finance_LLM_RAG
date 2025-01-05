import os, tempfile
from pathlib import Path
import logging
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from evaluate import RAG
logger = logging.getLogger(__name__)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
model_name = 'gpt-3.5-turbo'
category = "insurance"


import streamlit as st

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Streamlit Showcase: Unleashing the Power of RAG and LangChain")
mode = st.sidebar.radio(
    "LLM type：",
    ('Your own LLM', 'openAI'))
if mode == 'Your own LLM':
    openai_api_base = st.sidebar.text_input('URL:', type='default')
    openai_api_key = 'None'
elif mode == 'openAI':
    openai_api_base = st.sidebar.text_input('api_base:', type='password')
    openai_api_key = st.sidebar.text_input('key:', type='password')



def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(texts, embedding=embeddings,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def define_llm():
    llm = ChatOpenAI(model=model_name,openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    return llm

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = define_llm(),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def query_llm_direct(query):
    llm = define_llm()
    llm_chain = add_prompt(llm, query)
    result = llm_chain.invoke({"query": query})
    result = result['text']
    st.session_state.messages.append((query, result))
    return result

def load_vector_store(vector_store_path):
    logger.info("processing load_vector_store...")
    vector_db = {}
    
    db_store_folder = vector_store_path

    #finance
    db_finance_path = db_store_folder + "/db_finance/"
    vector_db['finance'] = finance_vector_store = FAISS.load_local(
        db_finance_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


    #insurance
    db_finance_path = db_store_folder + "/db_insurance/"
    vector_db['insurance'] = insurance_vector_store = FAISS.load_local(
        db_finance_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    #faq
    return vector_db


def query_llm_answer(query, category,db_vector):

    llm = define_llm()
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Answer in mandarin chinese:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    if category =="finance":
            docsearch = db_vector['finance']
    elif category =="insurance":
            docsearch = db_vector['insurance']
    else:
        logger.warnning("category not exist")
        docsearch = db_vector['insurance']
        #docsearch = FAISS.from_documents(text_chunks, embeddings)

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=docsearch.as_retriever(), 
            verbose=True,
            chain_type_kwargs=chain_type_kwargs
    )
    result = qa.invoke(query)



    result = result['result']
    st.session_state.messages.append((query, result))
    return result



def add_prompt(llm, query):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    init_Prompt = """
    you are helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision. \
    Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
    relevant, and concise: \
    {query}
    """
    
    input_prompt = PromptTemplate(input_variables=["query"], template=init_Prompt)

    return LLMChain(prompt=input_prompt, llm=llm)

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)


def process_documents():
    if not openai_api_base or not openai_api_key:
        st.warning(f"Please provide information about LLM model.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def insurance_domain_select():
    category = "insurance"

def finance_domain_select():
    category = "finance"

def boot():
    #
    input_fields()
    #
    left, middle, right = st.columns(3)
    left.button("Submit Documents", on_click=process_documents)
    middle.button("Finance QA ", on_click=finance_domain_select)
    right.button("insurance QA ", on_click=insurance_domain_select)
    #st.button("Submit Documents", on_click=process_documents)
    #
    #st.button("Finance QA ", on_click=finance_domain_select)
    #st.button("insurance QA ", on_click=insurance_domain_select)    

    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    vector_store_path = "./vector_stores"
    if query := st.chat_input():
        st.chat_message("human").write(query)
        db_vector = load_vector_store(vector_store_path)
        
        if "retriever" in st.session_state:
            response = query_llm(st.session_state.retriever, query)
            #response = query_llm_answer(st.session_state.retriever, query,category,db_vector)
        else:
            #response = query_llm_direct(query)
            response = query_llm_answer(query,category,db_vector)

        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
    