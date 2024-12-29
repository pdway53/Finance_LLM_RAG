import torch
import json
import logging
import os
from pathlib import Path
import sys
import argparse
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import pdfplumber  
from rank_bm25 import BM25Okapi  
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.docstore.document import Document
import cohere  
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from BM25Retrieve import BM25Retriever
from dataclasses import dataclass
from langchain.prompts import PromptTemplate
from text_processing import regex_processing

model = "GanymedeNil/text2vec-large-chinese"  
embeddings = HuggingFaceEmbeddings(model_name = model)
API_SECRET_KEY = "XXXXXX" #"OPENAI_API KEY";
os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
Cohere_API = "XXXXXX" #Cohere_API KEY"  
co = cohere.Client( api_key =Cohere_API ) 
model_name = 'gpt-3.5-turbo'


logger = logging.getLogger(__name__)

def read_pdf(pdf_loc, page_infos: list = None):

    pdf = pdfplumber.open(pdf_loc)  
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  
        text = page.extract_text()  
        if text:
            pdf_text += text
    pdf.close()  

    return pdf_text  


def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  
    return corpus_dict


def validate_train_groundtruth(args,ans_arr):

    with open(args.qa_gt_path, 'rb') as f_s:
        data = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = data["ground_truths"]
    
    total_question_num = len(key_to_source_dict)
    qid_arr = []
    retrieve_arr = []
    category_arr = []
    predict_length = len(ans_arr)
    idx = 0 
    match = 0 
    insurance_acc = 0
    finance_acc = 0
    faq_acc = 0
    for q in key_to_source_dict:
        print(q["qid"])
        print(q["retrieve"])
        print(q["category"])
        retrieve_arr.append(q["retrieve"])

        if idx < predict_length and ans_arr[idx] ==q["retrieve"]:
            match = match + 1

        if idx < predict_length and ans_arr[idx] ==q["retrieve"] and q["category"] == "insurance":
            insurance_acc = insurance_acc + 1
        
        if idx < predict_length and ans_arr[idx] ==q["retrieve"] and q["category"] == "finance":
            finance_acc = finance_acc + 1
        if idx < predict_length and ans_arr[idx] ==q["retrieve"] and q["category"] == "faq":
            faq_acc = faq_acc + 1
        idx = idx + 1

    print("match : ", match)
    print("insurance_acc : ", insurance_acc)
    print("finance_acc : ", finance_acc)
    print("faq_acc : ", faq_acc)

@dataclass
class RAGConfig:
    """
    Attributes:
        cache_dir (str): CKIP和向量存儲的緩存目錄
        vector_store_path (str): 向量存儲的保存路徑  
        model_name (str): 使用的OpenAI模型名稱
        temperature (float): OpenAI模型的temperature參數
        chunk_size (int): 文本分塊大小
        chunk_overlap (int): 文本分塊重疊大小
    """
    # 必需參數
    cache_dir: str
    vector_store_path: str
    model_name: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    rebuild_stores: bool = False  

    


def setup_argparser(args):    
    
    args = parser.parse_args()
    
    config = RAGConfig( cache_dir=args.cache_dir, vector_store_path=args.vector_store_path,
        #device=args.device, 
        model_name=args.model_name,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild_stores=args.rebuild_stores
    )

    return config


class RAG:
    def __init__(self, args, config: RAGConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature #config.temperature
        )
        #hybrid search
        #self.bm25_retriever = BM25Retriever(self.query_processor.ckip_processor)


    def process_documents(self, doc_path: str) -> Dict[str, Dict[int, str]]:
        source_path_insurance = os.path.join(args.source_path, 'insurance5')  # 設定參考資料路徑
        corpus_dict_insurance = load_data(source_path_insurance)
        source_path_finance = os.path.join(args.source_path, 'finance5')  # 設定參考資料路徑
        corpus_dict_finance = load_data(source_path_finance)

    def coherence_rerank(self, args, qs_ref)-> tuple[ List, Dict[str, int] ] :
        answer_dict = {"answers": []}  # 初始化字典
        ans_arr = []
        source_path_insurance = os.path.join(args.source_path, 'insurance/')  # 設定參考資料路徑
        source_path_finance = os.path.join(args.source_path, 'finance/')  # 設定參考資料路徑
        with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
            key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
    
        for q_dict in qs_ref['questions']:
            print(q_dict['source'])
            if q_dict['category'] == 'faq':                
                corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}                
                retrieved = self.llm_embedding_retrieve_faq(args,q_dict['query'], q_dict['source'], corpus_dict_faq)  
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
                ans_arr.append(retrieved)   
            else: #q_dict['category'] == 'finance':
                retrieved = self.llm_embedding_rerank_finance(args,q_dict['query'], q_dict['source'], q_dict)
                answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
                ans_arr.append(retrieved)
        
        return ans_arr , answer_dict
    
    def llm_embedding_rerank_finance(self, args,qs, source, q_dict)->int:

        rerank_docs = []
        source_path_insurance = os.path.join(args.source_path, 'insurance/')  # 設定參考資料路徑
        source_path_finance = os.path.join(args.source_path, 'finance/')  # 設定參考資料路徑
        if q_dict['category'] == 'insurance':
            loader = PyPDFLoader(source_path_finance + str(source[0]) + ".pdf" )
        elif q_dict['category'] == 'insurance':
            loader = PyPDFLoader(source_path_insurance + str(source[0]) + ".pdf" )
  
        for idx in source:
            if q_dict['category'] == 'finance':
                file = source_path_finance + str(idx) + ".pdf"
            elif q_dict['category'] == 'insurance':
                file = source_path_insurance + str(idx) + ".pdf"
            elif q_dict['category'] == 'faq':
                continue
        
            loader_pdf = PyPDFLoader(file)
            docs_list = loader_pdf.load()    
            text_splitter= RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size,  chunk_overlap=self.config.chunk_overlap)
            text_chunks = text_splitter.split_documents(docs_list)  
            rerank_docs.append(docs_list[0].page_content)
        
        # Create the retriever and set it to return a good amount of chunks 
        results = co.rerank(model="rerank-multilingual-v3.0", query= qs, documents=rerank_docs, top_n=5, return_documents=False)        
        rerank_indexes = [x.index for x in results.results]        
        best_best_indexes = [source[i] for i in rerank_indexes]
        
        return best_best_indexes[0]


    def llm_embedding_retrieve_faq(self, args,qs, source, corpus_dict_faq):
        logger.info("Creating FAISS index...")    

        rerank_docs = []
        for key, value in corpus_dict_faq.items():

            docs_list = value
            doc =  Document(page_content=value, metadata={"source": key})
            rerank_docs.append(doc)

        docs_list = docs_list       
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        new_db = FAISS.from_documents(rerank_docs, embeddings)        
        docs_mmr = new_db.max_marginal_relevance_search(qs,k=2)
        idx = docs_mmr[0].metadata['source']          
        
        return idx
    

    def LLM_answer(self, args, qs_ref,answers_retrieve):

        answerQA_dict = {"answers": []}  # 初始化字典
        question = qs_ref['questions']
        source_path_insurance = os.path.join(args.source_path, 'insurance/')  # 設定參考資料路徑
        source_path_finance = os.path.join(args.source_path, 'finance/')  # 設定參考資料路徑

        for retrieve in answers_retrieve['answers']:
            qid = retrieve['qid']
            retrieve_idx = retrieve['retrieve']
            logger.info("qid : {}".format(qid))
            logger.info("retrieve_idx : {}".format(retrieve_idx))

            category = question[qid-1]['category']
            query = question[qid-1]['query']            

            if category == 'finance':
                file = source_path_finance + str(retrieve_idx) + ".pdf"
            elif category == 'insurance':
                file = source_path_insurance + str(retrieve_idx) + ".pdf"
            elif category == 'faq':
                continue
            

            loader_pdf = PyPDFLoader(file)
            docs_list = loader_pdf.load()             
            text_splitter= RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size,  chunk_overlap=self.config.chunk_overlap)
            for doc in docs_list:
                doc.page_content = regex_processing(doc.page_content)
            
            text_chunks = text_splitter.split_documents(docs_list)        
             
            docsearch = FAISS.from_documents(text_chunks, embeddings)
            doc_text = [text for text in text_chunks]
            

            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Answer in mandarin chinese:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=self.llm, 
                chain_type="stuff", 
                retriever=docsearch.as_retriever(), 
                verbose=True,
                chain_type_kwargs=chain_type_kwargs
            )
            response = qa.invoke(query)

            logger.info("QA respnse : {}".format(response))


            answerQA_dict['answers'].append({"qid": query, "retrieve": response['result']})
        return answerQA_dict





def main(args):

    # initial RAG
    config = setup_argparser(args)
    logger.info("Initializing RAG system...")
    rag_db_system = RAG(args , config)

    vector_stores = {}
    Documents = {}
    #load vector
    #store = rag_db_system.process_documents(args.source_path)

    #load q
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    if args.rerank_with_LLM:
        ans, answers_retrieve = rag_db_system.coherence_rerank(args, qs_ref)

    else:
        #retreive predicted source file     
        with open(args.output_path, 'rb') as f:
            answers_retrieve = json.load(f)
        
    #將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answers_retrieve, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    if args.llm_answer:
        qa_ans = rag_db_system.LLM_answer(args,qs_ref,answers_retrieve)

        with open(args.output_qa_path, 'w', encoding='utf8') as f:
            json.dump(qa_ans, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    if args.test_traindata:
        validate_train_groundtruth(args,ans)


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--qa_gt_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    parser.add_argument('--output_qa_path', type=str, required=True, help='輸出LLM 回答結果路徑')  # 答案輸出的路徑
    parser.add_argument('--rerank_with_LLM', type=str,default = 0 , help='rank rag files')  # 答案輸出的路徑
    parser.add_argument('--test_traindata', type=str, default = 0 , help='test with ground true answer')  # 答案輸出的路徑
    parser.add_argument('--llm_answer', type=str, default = 0 , help='LLM answer question with rag source')  # 答案輸出的路徑
    parser.add_argument('--cache_dir', type=str,   default='ckip_cache',  help='Directory for CKIP cache' )
    parser.add_argument('--device',  type=int,  default=0,  help='GPU device ID (-1 for CPU)' )
    parser.add_argument('--temperature', type=float,  default=0.5,  help='Temperature for OpenAI model')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Size of text chunks for processing')
    parser.add_argument('--chunk_overlap', type=int, default=200, help='chunk_overlap')
    parser.add_argument('--rebuild-stores',  action='store_true',  help='Force rebuild of vector stores' )
    parser.add_argument('--vector_store_path', type=str,  default='vector_stores', help='Directory for vector stores')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo',  help='model_name' ) 


    args = parser.parse_args()  # 解析參數
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)