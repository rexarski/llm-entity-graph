from typing import Optional
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class EmbeddingModel:
    _instance: Optional[HuggingFaceBgeEmbeddings] = None

    @classmethod
    def get_instance(cls) -> HuggingFaceBgeEmbeddings:
        """获取嵌入模型单例"""
        if cls._instance is None:
            print("Initializing embedding model...")
            # cls._instance = HuggingFaceBgeEmbeddings(
            #     model_name="BAAI/bge-m3",
            #     model_kwargs={"device": "cpu"},
            #     encode_kwargs={"normalize_embeddings": True},
            #     query_instruction="",
            # )
            cls._instance = HuggingFaceBgeEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                model_kwargs={"device": "cpu", "trust_remote_code": True},
                encode_kwargs={"normalize_embeddings": True},
                query_instruction="",
            )
        return cls._instance

    @classmethod
    def cleanup(cls) -> None:
        """清理嵌入模型（如果需要）"""
        cls._instance = None
