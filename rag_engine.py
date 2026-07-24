import logging
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAG_Engine")


class RAGEngine:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        """RAG engine's initialization"""
        logger.info(f"Loading embedding model ({model_name})...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = len(self.encoder.encode("test"))
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: list[str] = []
        logger.info(f"RAG engine initialized. Vector dimension: {self.dimension}")

    def add_documents(self, texts: list[str]) -> None:
        """Takes list of strings, turns them into vectors, and saves inside FAISS."""
        valid_texts = [text.strip() for text in texts if text and text.strip()]

        if not valid_texts:
            logger.warning("Attempted to add an empty list of documents")
            return

        logger.info(f"Generating embeddings for {len(valid_texts)} fragments")

        e5_formatted_texts = [f"passage: {text}" for text in valid_texts]

        embeddings = self.encoder.encode(
            e5_formatted_texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

        self.index.add(embeddings)
        self.documents.extend(valid_texts)

        logger.info(
            f"Successfully added {len(valid_texts)} fragments. "
            f"Total documents in database: {self.index.ntotal}"
        )

    def search(self, query: str, top_k: int = 2, threshold: float = 0.78) -> list[str]:
        """Finds top k most matching text fragmets for query"""
        if self.index.ntotal == 0:
            logger.warning(f"The database is empty! Returned an empty list")
            return []

        logger.info(
            f"Searching for matching contexts in FAISS (top_k={top_k}) for: {query}"
        )

        e5_formatted_query = f"query: {query}"

        query_vector = self.encoder.encode(
            [e5_formatted_query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= threshold:
                results.append(self.documents[idx])

        logger.info(f"Found {len(results)} matches")
        logger.info(f"{results}")
        return results


if __name__ == "__main__":
    logger.info("=== Launching the RAG engine's standalone test")
    rag = RAGEngine()

    sample_knowledge = [
        (
            "Root Rule (LAG / LOZH): In Russian grammar, 'A' is written before 'G'"
            " (polagat'), and 'O' is written before 'ZH' (polozhit'). Exception:"
            " polog."
        ),
        (
            "NewUU University (New Uzbekistan University) is located in Tashkent"
            " and specializes in training AI & Robotics engineers."
        ),
        (
            "Aya Assistant is built on top of the local Gemma model,"
            " SentenceTransformers embeddings, and a FastAPI backend."
        ),
        (
            "Retrieval-Augmented Generation (RAG) combines semantic vector"
            " search with Large Language Models to reduce hallucinations."
        ),
    ]

    rag.add_documents(sample_knowledge)

    test_queries = [
        "Where is NewUU located and what do they study?",
        "Как работать с корнями лаг и лож?",
        "What tech stack is used for Aya?",
    ]

    print("\n================ RAG SEARCH TESTS ================\n")
    for query in test_queries:
        found_context = rag.search(query, top_k=1)
        result = found_context[0] if found_context else "Nothing found"
        print(f"Query: {query}")
        print(f"Result: {result}\n" + "-" * 50)
    print("==================================================\n")
