import os
from typing import List, Optional
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
                
                # Extract text from metadata with better field detection
                text = ""
                metadata_keys = []
                if hasattr(item, 'metadata') and item.metadata:
                    metadata_keys = list(item.metadata.keys())
                    # Try multiple possible field names for text content
                    for text_field in ["text", "content", "chunk_text", "chunk", "body"]:
                        if text_field in item.metadata:
                            text = item.metadata[text_field]
                            self.logger.info(f"Match {i}: found text in field '{text_field}' (length: {len(text)})")
                            break
                    
                    # If still no text found, log all metadata for debugging
                    if not text:
                        self.logger.info(f"Match {i}: no text found in any field. Full metadata: {item.metadata}")
                    
                    self.logger.info(f"Match {i}: metadata keys = {metadata_keys}")
                elif isinstance(item, dict) and "metadata" in item:
                    metadata_keys = list(item["metadata"].keys())
                    for text_field in ["text", "content", "chunk_text", "chunk", "body"]:
                        if text_field in item["metadata"]:
                            text = item["metadata"][text_field]
                            self.logger.info(f"Match {i}: found text in field '{text_field}' (length: {len(text)})")
                            break
                    
                    if not text:
                        self.logger.info(f"Match {i}: no text found in any field. Full metadata: {item['metadata']}")
                    
                    self.logger.info(f"Match {i}: metadata keys = {metadata_keys}")
                else:
                    self.logger.info(f"Match {i}: no metadata found")
                    continue
                
                # VERY aggressive threshold - accept almost anything for debugging
                if score > 0.01:  # Accept very low scores for debugging
                    if text:
                        chunks.append(text)
                        self.logger.info(f"Match {i}: added to chunks (length: {len(text)})")
                    else:
                        self.logger.info(f"Match {i}: no text content found despite having metadata")
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
        """Fallback method with very low threshold and more results"""
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
            
            self.logger.info(f"Fallback: Found {len(matches)} matches")
            
            chunks = []
            for i, item in enumerate(matches):
                score = getattr(item, 'score', 0)
                self.logger.info(f"Fallback Match {i}: score = {score}")
                
                # Accept ANYTHING with a score > 0 for debugging
                if score <= 0:
                    continue
                
                text = ""
                if hasattr(item, 'metadata') and item.metadata:
                    # Log full metadata for debugging
                    self.logger.info(f"Fallback Match {i}: Full metadata = {item.metadata}")
                    
                    # Try all possible text fields
                    for text_field in ["text", "content", "chunk_text", "chunk", "body", "data"]:
                        if text_field in item.metadata and item.metadata[text_field]:
                            text = item.metadata[text_field]
                            self.logger.info(f"Fallback Match {i}: found text in '{text_field}' (length: {len(text)})")
                            break
                elif isinstance(item, dict) and "metadata" in item:
                    self.logger.info(f"Fallback Match {i}: Full metadata = {item['metadata']}")
                    
                    for text_field in ["text", "content", "chunk_text", "chunk", "body", "data"]:
                        if text_field in item["metadata"] and item["metadata"][text_field]:
                            text = item["metadata"][text_field]
                            self.logger.info(f"Fallback Match {i}: found text in '{text_field}' (length: {len(text)})")
                            break
                
                if text:
                    chunks.append(text)
                    self.logger.info(f"Fallback Match {i}: added to chunks")
                else:
                    self.logger.info(f"Fallback Match {i}: no text content found")
            
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
                top_k=100,  # Increased to get more comprehensive stats
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

    def inspect_metadata_structure(self) -> dict:
        """Inspect the actual metadata structure in the namespace"""
        try:
            # Use a simple query to get sample records
            dummy_vector = [0.0] * 384  # Assuming 384-dimensional embeddings
            result = self.index.query(
                vector=dummy_vector,
                top_k=3,
                include_metadata=True,
                namespace=self.namespace
            )
            
            inspection = {
                "namespace": self.namespace,
                "total_matches": 0,
                "sample_records": []
            }
            
            if hasattr(result, 'matches'):
                matches = result.matches
            else:
                matches = getattr(result, 'data', [])
            
            inspection["total_matches"] = len(matches)
            
            for i, match in enumerate(matches):
                record_info = {
                    "index": i,
                    "score": getattr(match, 'score', 0),
                    "id": getattr(match, 'id', 'unknown'),
                    "metadata_keys": [],
                    "metadata_sample": {}
                }
                
                if hasattr(match, 'metadata') and match.metadata:
                    record_info["metadata_keys"] = list(match.metadata.keys())
                    # Sample a few metadata fields
                    for key in record_info["metadata_keys"][:5]:  # First 5 keys
                        value = match.metadata[key]
                        if isinstance(value, str) and len(value) > 100:
                            record_info["metadata_sample"][key] = value[:100] + "..."
                        else:
                            record_info["metadata_sample"][key] = value
                
                inspection["sample_records"].append(record_info)
            
            return inspection
            
        except Exception as e:
            self.logger.error(f"Failed to inspect metadata structure: {e}")
            return {
                "namespace": self.namespace,
                "error": str(e)
            }

    def debug_all_namespaces(self) -> dict:
        """Debug method to check what namespaces exist (if supported by Upstash)"""
        try:
            # Try to query without namespace to see if there's any data at all
            dummy_vector = [0.0] * 384
            result_no_ns = self.index.query(
                vector=dummy_vector,
                top_k=10,
                include_metadata=True
                # No namespace parameter
            )
            
            debug_info = {
                "current_namespace": self.namespace,
                "data_without_namespace": False,
                "total_without_namespace": 0,
                "sample_metadata": []
            }
            
            if hasattr(result_no_ns, 'matches') and result_no_ns.matches:
                debug_info["data_without_namespace"] = True
                debug_info["total_without_namespace"] = len(result_no_ns.matches)
                
                # Collect sample metadata to see what's available
                for match in result_no_ns.matches[:3]:  # First 3 matches
                    if hasattr(match, 'metadata') and match.metadata:
                        debug_info["sample_metadata"].append({
                            "keys": list(match.metadata.keys()),
                            "has_text": "text" in match.metadata,
                            "has_source": "source" in match.metadata,
                        })
            
            return debug_info
            
        except Exception as e:
            self.logger.error(f"Debug all namespaces failed: {e}")
            return {
                "current_namespace": self.namespace,
                "error": str(e)
            }

    def answer_with_debug(self, question: str) -> dict:
        """Answer with comprehensive debugging information"""
        self.logger.info(f"Processing question with debug for namespace {self.namespace}: {question}")
        
        # Get namespace stats first
        namespace_stats = self.get_namespace_stats()
        debug_info = self.debug_all_namespaces()
        
        question_vector = self.embed_question(question)
        chunks = self.get_context_chunks(question_vector, top_k=5)

        result = {
            "question": question,
            "namespace": self.namespace,
            "namespace_stats": namespace_stats,
            "debug_info": debug_info,
            "chunks_found": len(chunks),
            "answer": ""
        }

        if not chunks:
            self.logger.warning(f"No relevant context found for question in namespace {self.namespace}: {question}")
            # Try with a lower threshold or different approach
            chunks = self.get_context_chunks_fallback(question_vector, top_k=10)
            result["fallback_chunks_found"] = len(chunks)
            
            if not chunks:
                result["answer"] = f"Sorry, I couldn't find any relevant information to answer your question in the knowledge base for user {self.user_id} and bot {self.bot_id}. Please make sure the knowledge base contains information about your query."
                return result

        context = "\n\n".join(chunks)
        result["context_length"] = len(context)
        
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
            result["answer"] = response.choices[0].message.content.strip()
            self.logger.info(f"Generated answer for namespace {self.namespace} (length: {len(result['answer'])})")
        except Exception as e:
            self.logger.error(f"OpenAI request failed for namespace {self.namespace}: {e}")
            result["answer"] = "Something went wrong while getting the answer from OpenAI."
            result["openai_error"] = str(e)
        
        return result


# Usage example:
# qa_system = VectorSearchQA(user_id="user123", bot_id="bot456")
# answer = qa_system.answer("What is the main topic discussed in the documents?")
# print(answer)