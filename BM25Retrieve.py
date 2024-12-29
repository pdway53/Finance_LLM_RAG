import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Optional, Union
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """BM25檢索器"""

    def __init__(self, ckip_processor, use_jieba: bool = False):
        self.ckip_processor = ckip_processor
        self.use_jieba = use_jieba
        if use_jieba:
            import jieba
            self.jieba = jieba
            logger.info("Using Jieba for tokenization")
        else:
            logger.info("Using CKIP for tokenization")

        self.corpus_dict = {
            'finance': {},
            'insurance': {},
            'faq': {}
        }

    def _tokenize(self, text: str) -> List[str]:
        """分詞方法"""
        if self.use_jieba:
            return list(self.jieba.cut_for_search(text))
        else:
            return self.ckip_processor.segment_parallel([text])[0].split()

    def create_store(self, category: str, texts: List[str], file_ids: List[int]) -> None:
        """保存文檔集合"""
        try:
            self.corpus_dict[category] = {
                file_id: text
                for file_id, text in zip(file_ids, texts)
            }
            logger.info(f"Saved {len(texts)} documents for category {category}")
        except Exception as e:
            logger.error(f"Error saving documents for {category}: {e}")
            raise

    def retrieve(self, query: str, source: List[str], category: str) -> int:
        """執行BM25檢索"""
        try:
            if category not in self.corpus_dict:
                raise ValueError(f"No documents found for category: {category}")

            # 只取source中的文檔
            filtered_corpus = []
            valid_source_ids = []

            for file_id in source:
                doc_id = int(file_id)
                if doc_id in self.corpus_dict[category]:
                    filtered_corpus.append(self.corpus_dict[category][doc_id])
                    valid_source_ids.append(doc_id)

            source_ids = [int(s) for s in source]

            if not filtered_corpus:
                logger.warning("No valid documents found in source")
                return source_ids[0]

            # 對篩選後的文檔進行分詞
            tokenized_corpus = [self._tokenize(doc) for doc in filtered_corpus]

            # 建立BM25模型
            bm25 = BM25Okapi(tokenized_corpus)

            # 對查詢進行分詞
            tokenized_query = self._tokenize(query)

            # 使用 get_top_n 取得最相關文檔
            top_docs = bm25.get_top_n(
                tokenized_query,
                filtered_corpus,
                n=1
            )

            if not top_docs:
                logger.warning("No results from BM25")
                return source_ids[0]

            best_doc = top_docs[0]

            # 找回文檔ID
            for i, doc_text in enumerate(filtered_corpus):
                if doc_text == best_doc:
                    retrieved_id = valid_source_ids[i]
                    logger.info(f"BM25 selected document {retrieved_id}")
                    return retrieved_id

            logger.warning("Could not map result back to document ID")
            return source_ids[0]

        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return source_ids[0] if source_ids else -1

    def create_stores_from_documents(self, documents: Dict[str, Dict[int, str]]) -> None:
        """從文檔集合創建所有類別的存儲"""
        for category, docs in documents.items():
            try:
                texts = list(docs.values())
                file_ids = [int(id_) for id_ in docs.keys()]
                self.create_store(category, texts, file_ids)
            except Exception as e:
                logger.error(f"Failed to create store for {category}: {e}")