import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from openai import OpenAI
from botforge.core.config import settings
from botforge.core.logger import log


class VectorSearchQA:
    def __init__(self, user_id: str, bot_id: str, model_name: str = "gpt-3.5-turbo"):
        self.logger = log
        self.user_id = user_id
        self.bot_id = bot_id
        self.namespace = f"{user_id}_{bot_id}"
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.index = Index(url=settings.upstash_url, token=settings.upstash_token)
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = model_name

        self.logger.info(f"Initialized VectorSearchQA with namespace: {self.namespace} using model: {self.model_name}")

    def embed_question(self, question: str) -> List[float]:
        return self.embedder.encode(question, normalize_embeddings=True).tolist()

    def query_vector_index(self, vector: List[float], top_k: int = 5) -> List[str]:
        try:
            result = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = getattr(result, 'matches', getattr(result, 'data', []))
            sources = [
                item.metadata["source"]
                for item in matches
                if hasattr(item, 'metadata') and item.metadata and "source" in item.metadata
            ]
            self.logger.info(f"Found {len(sources)} sources in namespace {self.namespace}")
            return sources
        except Exception as e:
            self.logger.error(f"Vector search failed for namespace {self.namespace}: {e}")
            return []

    def get_context_chunks(self, vector: List[float], top_k: int = 5) -> List[str]:
        try:
            result = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = getattr(result, 'matches', getattr(result, 'data', []))
            self.logger.info(f"Found {len(matches)} matches in namespace {self.namespace}")
            chunks = []

            for i, item in enumerate(matches):
                score = getattr(item, 'score', 0)
                self.logger.info(f"Match {i}: score = {score}")
                text = ""

                if hasattr(item, 'metadata') and item.metadata:
                    for field in ["text", "content", "chunk_text", "chunk", "body"]:
                        if field in item.metadata:
                            text = item.metadata[field]
                            self.logger.info(f"Match {i}: found text in field '{field}'")
                            break
                elif isinstance(item, dict) and "metadata" in item:
                    for field in ["text", "content", "chunk_text", "chunk", "body"]:
                        if field in item["metadata"]:
                            text = item["metadata"][field]
                            break

                if score > 0.01 and text:
                    chunks.append(text)

            self.logger.info(f"Returning {len(chunks)} chunks from namespace {self.namespace}")
            return chunks

        except Exception as e:
            self.logger.error(f"Failed to get context chunks from namespace {self.namespace}: {e}")
            return []

    def answer(self, question: str) -> str:
        self.logger.info(f"Processing question for namespace {self.namespace}: {question}")
        question_vector = self.embed_question(question)
        chunks = self.get_context_chunks(question_vector)

        if not chunks:
            self.logger.warning(f"No context found. Trying fallback.")
            chunks = self.get_context_chunks_fallback(question_vector)
            if not chunks:
                return "Sorry, I couldn't find any relevant information to answer your question."

        context = "\n\n".join(chunks)
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant who answers based on context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI request failed: {e}")
            return "Something went wrong while getting the answer from OpenAI."

    def get_context_chunks_fallback(self, vector: List[float], top_k: int = 10) -> List[str]:
        try:
            result = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = getattr(result, 'matches', getattr(result, 'data', []))
            chunks = []

            for i, item in enumerate(matches):
                score = getattr(item, 'score', 0)
                if score <= 0:
                    continue

                text = ""
                if hasattr(item, 'metadata') and item.metadata:
                    for field in ["text", "content", "chunk_text", "chunk", "body", "data"]:
                        if field in item.metadata and item.metadata[field]:
                            text = item.metadata[field]
                            break
                elif isinstance(item, dict) and "metadata" in item:
                    for field in ["text", "content", "chunk_text", "chunk", "body", "data"]:
                        if field in item["metadata"] and item["metadata"][field]:
                            text = item["metadata"][field]
                            break

                if text:
                    chunks.append(text)

            return chunks
        except Exception as e:
            self.logger.error(f"Fallback method failed: {e}")
            return []

    def get_namespace_stats(self) -> dict:
        try:
            dummy_vector = [0.0] * 384
            result = self.index.query(
                vector=dummy_vector,
                top_k=100,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = getattr(result, 'matches', getattr(result, 'data', []))
            return {
                "namespace": self.namespace,
                "user_id": self.user_id,
                "bot_id": self.bot_id,
                "has_data": bool(matches),
                "total_matches": len(matches)
            }
        except Exception as e:
            self.logger.error(f"Failed to get namespace stats: {e}")
            return {
                "namespace": self.namespace,
                "error": str(e)
            }

    def inspect_metadata_structure(self) -> dict:
        try:
            dummy_vector = [0.0] * 384
            result = self.index.query(
                vector=dummy_vector,
                top_k=3,
                include_metadata=True,
                namespace=self.namespace
            )
            matches = getattr(result, 'matches', getattr(result, 'data', []))
            samples = []

            for i, match in enumerate(matches):
                metadata = getattr(match, 'metadata', {})
                sample = {
                    "index": i,
                    "score": getattr(match, 'score', 0),
                    "id": getattr(match, 'id', 'unknown'),
                    "metadata_keys": list(metadata.keys()) if metadata else [],
                    "metadata_sample": {k: (v[:100] + "...") if isinstance(v, str) and len(v) > 100 else v
                                        for k, v in list(metadata.items())[:5]} if metadata else {}
                }
                samples.append(sample)

            return {
                "namespace": self.namespace,
                "total_matches": len(matches),
                "sample_records": samples
            }
        except Exception as e:
            self.logger.error(f"Failed to inspect metadata structure: {e}")
            return {
                "namespace": self.namespace,
                "error": str(e)
            }

    def debug_all_namespaces(self) -> dict:
        try:
            dummy_vector = [0.0] * 384
            result = self.index.query(
                vector=dummy_vector,
                top_k=10,
                include_metadata=True
            )
            matches = getattr(result, 'matches', getattr(result, 'data', []))
            return {
                "current_namespace": self.namespace,
                "data_without_namespace": bool(matches),
                "total_without_namespace": len(matches),
                "sample_metadata": [
                    {
                        "keys": list(getattr(match, 'metadata', {}).keys()),
                        "has_text": "text" in getattr(match, 'metadata', {}),
                        "has_source": "source" in getattr(match, 'metadata', {})
                    } for match in matches[:3]
                ]
            }
        except Exception as e:
            self.logger.error(f"Debug all namespaces failed: {e}")
            return {
                "current_namespace": self.namespace,
                "error": str(e)
            }

    def answer_with_debug(self, question: str) -> dict:
        self.logger.info(f"Processing question with debug for namespace {self.namespace}: {question}")
        namespace_stats = self.get_namespace_stats()
        debug_info = self.debug_all_namespaces()
        question_vector = self.embed_question(question)
        chunks = self.get_context_chunks(question_vector)

        result = {
            "question": question,
            "namespace": self.namespace,
            "namespace_stats": namespace_stats,
            "debug_info": debug_info,
            "chunks_found": len(chunks),
            "answer": ""
        }

        if not chunks:
            chunks = self.get_context_chunks_fallback(question_vector)
            result["fallback_chunks_found"] = len(chunks)
            if not chunks:
                result["answer"] = "Sorry, I couldn't find any relevant information."
                return result

        context = "\n\n".join(chunks)
        prompt = (
            "You are a helpful assistant. Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant who answers based on context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            result["answer"] = response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI request failed: {e}")
            result["answer"] = "Something went wrong while getting the answer."
            result["openai_error"] = str(e)

        return result
