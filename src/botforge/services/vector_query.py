import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from botforge.core.config import settings
from botforge.core.logger import log
from botforge.core.redis import redis_client
from botforge.services.caching import store_query_response_context, get_query_response_context


class VectorQueryServiceError(Exception):
    """Base exception for VectorQueryService"""
    pass


class EmbeddingError(VectorQueryServiceError):
    """Exception raised when embedding generation fails"""
    pass


class VectorQueryError(VectorQueryServiceError):
    """Exception raised when vector database query fails"""
    pass


class OpenAIError(VectorQueryServiceError):
    """Exception raised when OpenAI API call fails"""
    pass


class InitializationError(VectorQueryServiceError):
    """Exception raised when service initialization fails"""
    pass


class VectorQueryService:
    def __init__(self):
        """Initialize the vector query service"""
        self.logger = log
        try:
            # Initialize OpenAI
            if not settings.openai_api_key:
                raise InitializationError("OpenAI API key not found in settings")
            
            openai.api_key = settings.openai_api_key
            self.openai_client = openai
            
            # Initialize sentence transformer (same as uploader)
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            
            # Validate Upstash settings
            if not settings.upstash_url or not settings.upstash_token:
                raise InitializationError("Upstash URL or token not found in settings")
            
            # Initialize Upstash Vector index
            self.index = Index(
                url=settings.upstash_url,
                token=settings.upstash_token
            )
            
            self.logger.info("Vector Query Service initialized successfully")
            
        except InitializationError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize Vector Query Service: {e}"
            self.logger.error(error_msg)
            raise InitializationError(error_msg) from e

    def create_namespace(self, user_id: str, bot_id: str) -> str:
        """Create namespace from user_id and bot_id"""
        if not user_id or not bot_id:
            raise ValueError("user_id and bot_id must be non-empty strings")
        
        namespace = f"{user_id}_{bot_id}"
        self.logger.debug(f"Created namespace: {namespace}")
        return namespace

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        if not text or not text.strip():
            raise ValueError("Text must be non-empty")
        
        try:
            self.logger.debug(f"Generating embedding for text: '{text[:50]}...'")
            embedding = self.model.encode(text, normalize_embeddings=True)
            result = embedding.tolist()
            self.logger.debug(f"Generated embedding with {len(result)} dimensions")
            return result
        except Exception as e:
            error_msg = f"Error generating embedding: {e}"
            self.logger.error(error_msg)
            raise EmbeddingError(error_msg) from e

    async def query_vector_data(self, user_id: str, bot_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query vector database for relevant chunks"""
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        try:
            namespace = self.create_namespace(user_id, bot_id)
            self.logger.info(f"Querying namespace: {namespace} with query: '{query[:50]}...'")
            
            # Generate query embedding
            query_vector = self.embed_text(query)
            
            # Query the vector database
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_data=True
            )
            
            # Process results
            processed_results = []
            for result in results:
                try:
                    metadata = result.metadata or {}
                    data = result.data or {}
                    
                    processed_result = {
                        "id": result.id,
                        "score": result.score,
                        "content": data,
                        "source": metadata.get("source", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "metadata": metadata
                    }
                    processed_results.append(processed_result)
                except Exception as e:
                    self.logger.warning(f"Failed to process result {result.id}: {e}")
                    continue
            
            self.logger.info(f"Found {len(processed_results)} relevant chunks")
            return processed_results
            
        except (ValueError, EmbeddingError):
            raise
        except Exception as e:
            error_msg = f"Vector query failed: {e}"
            self.logger.error(error_msg)
            raise VectorQueryError(error_msg) from e

    async def generate_openai_response(
        self, 
        query: str, 
        client_id: str,
        context_chunks: List[Dict[str, Any]], 
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.7
    ) -> str:
        """Generate response using OpenAI with context"""
        if not query or not query.strip():
            raise ValueError("Query must be non-empty")
        
        if not client_id:
            raise ValueError("client_id must be provided")
        
        if temperature is not None and (temperature < 0 or temperature > 2):
            raise ValueError("Temperature must be between 0 and 2")
        
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        
        try:
            self.logger.debug(f"Generating OpenAI response for client: {client_id}")
            
            # Get chat history
            chat_history = await get_query_response_context(
                redis=redis_client,
                client_id=client_id 
            )
            
            # Build context text
            context_text = ""
            for i, chunk in enumerate(context_chunks):
                if not isinstance(chunk, dict) or 'content' not in chunk:
                    self.logger.warning(f"Invalid chunk format at index {i}")
                    continue
                    
                source = chunk.get('source', 'Unknown')
                content = chunk.get('content', '')
                context_text += f"[Source {i+1}: {source}]\n{content}\n\n"
            
            system_prompt = """You are a friendly AI assistant augmented with an Upstash Vector Store.
                            To help you answer the questions, a context will be provided. This context is generated by querying the vector store with the user question.
                            Answer the question at the end using only the information available in the context and chat history.
                            If the answer is not available in the chat history or context, do not answer the question and politely let the user know that you can only answer if the answer is available in context or the chat history.

                            -------------
                            Chat history:
                            {chat_history}
                            -------------
                            Context:
                            {context}
                            -------------
                            Helpful answer: """

            user_prompt = f"Question: {query}"

            # Make OpenAI API call
            self.logger.debug("Making OpenAI API call")
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt.format(context=context_text, chat_history=chat_history)},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            if not response.choices or not response.choices[0].message:
                raise OpenAIError("Invalid response from OpenAI API")
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store the conversation context
            await store_query_response_context(
                redis=redis_client,
                client_id=client_id,
                query=query,
                response=ai_response
            )
            
            self.logger.debug(f"Generated response with {len(ai_response)} characters")
            return ai_response
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"OpenAI API call failed: {e}"
            self.logger.error(error_msg)
            raise OpenAIError(error_msg) from e

    async def query_and_generate_response(
        self,
        user_id: str,
        client_id: str,
        bot_id: str,
        query: str,
        model: str = "gpt-3.5-turbo",
        top_k: int = 5,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.7
    ) -> Dict[str, Any]:
        """Complete query and response generation pipeline"""
        if not all([user_id, client_id, bot_id, query]):
            raise ValueError("user_id, client_id, bot_id, and query must all be provided")
        
        try:
            start_time = time.time()
            self.logger.info(f"Starting query pipeline for user: {user_id}, bot: {bot_id}")
            
            # Step 1: Query vector database
            vector_results = await self.query_vector_data(user_id, bot_id, query, top_k)
            
            if not vector_results:
                self.logger.info("No relevant chunks found in vector database")
                return {
                    "response": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "total_chunks_found": 0,
                    "execution_time": time.time() - start_time
                }
            
            # Step 2: Generate OpenAI response
            ai_response = await self.generate_openai_response(
                client_id=client_id,
                query=query,
                context_chunks=vector_results,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Step 3: Prepare response
            sources = []
            for chunk in vector_results:
                try:
                    content = chunk.get("content", "")
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    
                    sources.append({
                        "source": chunk.get("source", "Unknown"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "relevance_score": chunk.get("score", 0.0),
                        "content_preview": content_preview
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to process source information: {e}")
                    continue
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Query completed in {execution_time:.2f}s")
            
            return {
                "response": ai_response,
                "sources": sources,
                "total_chunks_found": len(vector_results),
                "execution_time": execution_time
            }
            
        except (ValueError, VectorQueryError, OpenAIError):
            raise
        except Exception as e:
            error_msg = f"Query and generation pipeline failed: {e}"
            self.logger.error(error_msg)
            raise VectorQueryServiceError(error_msg) from e

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        try:
            start_time = time.time()
            self.logger.info("Running health check")
            
            # Test Upstash connection
            upstash_status = False
            upstash_error = None
            try:
                # Try a simple embedding and query
                test_vector = self.embed_text("health check")
                test_result = self.index.query(
                    vector=test_vector,
                    top_k=1,
                    namespace="health_check"
                )
                upstash_status = True
                self.logger.debug("Upstash health check passed")
            except Exception as e:
                upstash_error = str(e)
                self.logger.warning(f"Upstash health check failed: {e}")
            
            # Test OpenAI connection
            openai_status = False
            openai_error = None
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                )
                openai_status = True
                self.logger.debug("OpenAI health check passed")
            except Exception as e:
                openai_error = str(e)
                self.logger.warning(f"OpenAI health check failed: {e}")
            
            overall_status = upstash_status and openai_status
            response_time = time.time() - start_time
            
            result = {
                "overall_status": overall_status,
                "upstash_status": upstash_status,
                "openai_status": openai_status,
                "timestamp": datetime.now().isoformat(),
                "response_time": response_time
            }
            
            # Include error details if any
            if upstash_error:
                result["upstash_error"] = upstash_error
            if openai_error:
                result["openai_error"] = openai_error
            
            self.logger.info(f"Health check completed in {response_time:.2f}s, overall status: {overall_status}")
            return result
            
        except Exception as e:
            error_msg = f"Health check failed: {e}"
            self.logger.error(error_msg)
            raise VectorQueryServiceError(error_msg) from e

    async def test_query(self, client_id: str = "test_client") -> Dict[str, Any]:
        """Run a test query"""
        try:
            start_time = time.time()
            self.logger.info("Running test query")
            
            # Use a generic test query
            test_query = "What is the main topic of the documents?"
            
            # Try to find any available namespace for testing
            test_response = await self.generate_openai_response(
                query=test_query,
                client_id=client_id,
                context_chunks=[{
                    "content": "This is a test document for API testing purposes.",
                    "source": "test_doc.txt",
                    "chunk_index": 0
                }],
                model="gpt-3.5-turbo",
                max_tokens=100
            )
            
            execution_time = time.time() - start_time
            
            result = {
                "test_query": test_query,
                "response": test_response,
                "execution_time": execution_time,
                "status": "success"
            }
            
            self.logger.info(f"Test query completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Test query failed: {e}"
            self.logger.error(error_msg)
            raise VectorQueryServiceError(error_msg) from e

    async def list_available_namespaces(self) -> List[str]:
        """List available namespaces (this is a simplified version)"""
        try:
            # Note: Upstash Vector doesn't have a direct way to list namespaces
            # This is a placeholder - you might need to maintain a registry
            # or use a different approach based on your setup
            
            self.logger.info("Listing namespaces (placeholder implementation)")
            self.logger.warning("This is a placeholder implementation - consider maintaining a namespace registry")
            return ["user123_bot456", "test_namespace"]
            
        except Exception as e:
            error_msg = f"Failed to list namespaces: {e}"
            self.logger.error(error_msg)
            raise VectorQueryServiceError(error_msg) from e

    async def get_namespace_stats(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """Get statistics for a namespace"""
        if not user_id or not bot_id:
            raise ValueError("user_id and bot_id must be provided")
        
        try:
            namespace = self.create_namespace(user_id, bot_id)
            self.logger.info(f"Getting stats for namespace: {namespace}")
            
            # Try to get some sample data to estimate size
            test_vector = self.embed_text("sample query")
            results = self.index.query(
                vector=test_vector,
                top_k=100,  # Get more results to estimate total
                namespace=namespace,
                include_metadata=True
            )
            
            # Calculate basic stats
            total_chunks = len(results)
            sources = set()
            total_content_length = 0
            
            for result in results:
                try:
                    metadata = result.get("metadata", {})
                    if "source" in metadata:
                        sources.add(metadata["source"])
                    if "content" in metadata:
                        total_content_length += len(str(metadata["content"]))
                except Exception as e:
                    self.logger.warning(f"Failed to process result metadata: {e}")
                    continue
            
            avg_chunk_size = total_content_length // total_chunks if total_chunks > 0 else 0
            
            result = {
                "namespace": namespace,
                "estimated_total_chunks": total_chunks,
                "unique_sources": len(sources),
                "sources": list(sources),
                "average_chunk_size": avg_chunk_size,
                "total_content_length": total_content_length
            }
            
            self.logger.info(f"Namespace stats: {total_chunks} chunks, {len(sources)} sources")
            return result
            
        except ValueError:
            raise
        except Exception as e:
            error_msg = f"Failed to get namespace stats: {e}"
            self.logger.error(error_msg)
            raise VectorQueryServiceError(error_msg) from e