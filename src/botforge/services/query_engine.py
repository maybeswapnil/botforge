from typing import List
from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from openai import OpenAI
from botforge.core.config import settings
from botforge.core.logger import log


class VectorSearchQA:
    def __init__(self, user_id: str, bot_id: str):
        self.logger = log
        self.user_id = user_id
        self.bot_id = bot_id
        self.namespace = f"{user_id}_{bot_id}"
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.index = Index(url=settings.upstash_url, token=settings.upstash_token)
        self.client = OpenAI(api_key=settings.openai_api_key)
        
        self.logger.info(f"Initialized VectorSearchQA with namespace: {self.namespace}")

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
            
            # Handle the QueryResult object properly
            if hasattr(result, 'matches'):
                matches = result.matches
            else:
                # Fallback if result structure is different
                matches = getattr(result, 'data', [])
            
            sources = []
            for item in matches:
                if hasattr(item, 'metadata') and item.metadata and "source" in item.metadata:
                    sources.append(item.metadata["source"])
                elif isinstance(item, dict) and "metadata" in item and "source" in item["metadata"]:
                    sources.append(item["metadata"]["source"])
            
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
            
            # Handle the QueryResult object properly
            if hasattr(result, 'matches'):
                matches = result.matches
            else:
                # Fallback if result structure is different
                matches = getattr(result, 'data', [])
            
            self.logger.info(f"Found {len(matches)} matches in namespace {self.namespace}")
            
            chunks = []
            for i, item in enumerate(matches):
                # Check if item has score attribute and log it
                score = getattr(item, 'score', 0)
                self.logger.info(f"Match {i}: score = {score}")
                
                # Extract text from metadata
                text = ""
                if hasattr(item, 'metadata') and item.metadata:
                    text = item.metadata.get("text", "")
                    self.logger.info(f"Match {i}: metadata keys = {list(item.metadata.keys())}")
                elif isinstance(item, dict) and "metadata" in item:
                    text = item["metadata"].get("text", "")
                    self.logger.info(f"Match {i}: metadata keys = {list(item['metadata'].keys())}")
                else:
                    self.logger.info(f"Match {i}: no metadata found")
                    continue
                
                # Lower the threshold temporarily for debugging
                if score > 0.3:  # Lowered from 0.7 to 0.3
                    if text:
                        chunks.append(text)
                        self.logger.info(f"Match {i}: added to chunks (length: {len(text)})")
                    else:
                        self.logger.info(f"Match {i}: no text in metadata")
                else:
                    self.logger.info(f"Match {i}: score too low ({score})")
            
            self.logger.info(f"Returning {len(chunks)} chunks from namespace {self.namespace}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get context chunks from namespace {self.namespace}: {e}")
            self.logger.error(f"Result type: {type(result) if 'result' in locals() else 'No result'}")
            if 'result' in locals():
                self.logger.error(f"Result attributes: {dir(result) if hasattr(result, '__dict__') else 'No attributes'}")
            return []

    def answer(self, question: str) -> str:
        self.logger.info(f"Processing question for namespace {self.namespace}: {question}")
        
        question_vector = self.embed_question(question)
        chunks = self.get_context_chunks(question_vector, top_k=5)

        if not chunks:
            self.logger.warning(f"No relevant context found for question in namespace {self.namespace}: {question}")
            # Try with a lower threshold or different approach
            chunks = self.get_context_chunks_fallback(question_vector, top_k=10)
            
            if not chunks:
                return f"Sorry, I couldn't find any relevant information to answer your question in the knowledge base for user {self.user_id} and bot {self.bot_id}. Please make sure the knowledge base contains information about your query."

        context = "\n\n".join(chunks)
        
        self.logger.info(f"Using context of length: {len(context)} from namespace {self.namespace}")

        prompt = (
            "You are a helpful assistant. Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant who answers based on context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
            self.logger.info(f"Generated answer for namespace {self.namespace} (length: {len(answer)})")
            return answer
        except Exception as e:
            self.logger.error(f"OpenAI request failed for namespace {self.namespace}: {e}")
            return "Something went wrong while getting the answer from OpenAI."

    def get_context_chunks_fallback(self, vector: List[float], top_k: int = 10) -> List[str]:
        """Fallback method with lower threshold and more results"""
        try:
            result = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace
            )
            
            if hasattr(result, 'matches'):
                matches = result.matches
            else:
                matches = getattr(result, 'data', [])
            
            chunks = []
            for item in matches:
                score = getattr(item, 'score', 0)
                if score <= 0.1:
                    continue
                
                text = ""
                if hasattr(item, 'metadata') and item.metadata:
                    text = item.metadata.get("text", "")
                elif isinstance(item, dict) and "metadata" in item:
                    text = item["metadata"].get("text", "")
                
                if text:
                    chunks.append(text)
            
            self.logger.info(f"Fallback method found {len(chunks)} chunks in namespace {self.namespace}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Fallback method failed for namespace {self.namespace}: {e}")
            return []

    def get_namespace_stats(self) -> dict:
        """Get statistics about the current namespace"""
        try:
            # Query with a dummy vector to get stats
            dummy_vector = [0.0] * 384  # Assuming 384-dimensional embeddings
            result = self.index.query(
                vector=dummy_vector,
                top_k=1,
                include_metadata=True,
                namespace=self.namespace
            )
            
            stats = {
                "namespace": self.namespace,
                "user_id": self.user_id,
                "bot_id": self.bot_id,
                "has_data": False,
                "total_matches": 0
            }
            
            if hasattr(result, 'matches') and result.matches:
                stats["has_data"] = True
                stats["total_matches"] = len(result.matches)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get namespace stats for {self.namespace}: {e}")
            return {
                "namespace": self.namespace,
                "user_id": self.user_id,
                "bot_id": self.bot_id,
                "has_data": False,
                "total_matches": 0,
                "error": str(e)
            }


# Usage example:
# qa_system = VectorSearchQA(user_id="user123", bot_id="bot456")
# answer = qa_system.answer("What is the main topic discussed in the documents?")
# print(answer)