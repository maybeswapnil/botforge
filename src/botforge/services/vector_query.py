import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from botforge.core.config import settings
from botforge.core.logger import log

class VectorQueryService:
    def __init__(self):
        """Initialize the vector query service"""
        try:
            # Initialize OpenAI
            openai.api_key = settings.openai_api_key
            self.openai_client = openai
            
            # Initialize sentence transformer (same as uploader)
            self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            
            # Initialize Upstash Vector index
            self.index = Index(
                url=settings.upstash_url,
                token=settings.upstash_token
            )
            
            log.info("✅ Vector Query Service initialized successfully")
            
        except Exception as e:
            log.error(f"❌ Failed to initialize Vector Query Service: {e}")
            raise

    def create_namespace(self, user_id: str, bot_id: str) -> str:
        """Create namespace from user_id and bot_id"""
        return f"{user_id}_{bot_id}"

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            log.error(f"Error generating embedding: {e}")
            raise

    async def query_vector_data(self, user_id: str, bot_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query vector database for relevant chunks"""
        try:
            namespace = self.create_namespace(user_id, bot_id)
            log.info(f"🔍 Querying namespace: {namespace} with query: '{query[:50]}...'")
            
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
            
            log.info(f"📊 Found {len(processed_results)} relevant chunks")
            return processed_results
            
        except Exception as e:
            log.error(f"Vector query failed: {e}")
            raise

    async def generate_openai_response(
        self, 
        query: str, 
        context_chunks: List[Dict[str, Any]], 
        model: str = "gpt-3.5-turbo",
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.7
    ) -> str:
        """Generate response using OpenAI with context"""
        try:
            # Prepare context from chunks
            context_text = ""
            for i, chunk in enumerate(context_chunks):
                context_text += f"[Source {i+1}: {chunk['source']}]\n{chunk['content']}\n\n"
            
            # Create the prompt
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer the user's question accurately and comprehensively. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Guidelines:
- Answer based primarily on the provided context
- Be specific and cite sources when possible
- If information is not in the context, clearly state that
- Provide helpful and detailed responses
- Format your response clearly and professionally"""

            user_prompt = f"Question: {query}"

            # Make OpenAI API call
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt.format(context=context_text)},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            log.error(f"OpenAI API call failed: {e}")
            raise

    async def query_and_generate_response(
        self,
        user_id: str,
        bot_id: str,
        query: str,
        model: str = "gpt-3.5-turbo",
        top_k: int = 5,
        max_tokens: Optional[int] = 1000,
        temperature: Optional[float] = 0.7
    ) -> Dict[str, Any]:
        """Complete query and response generation pipeline"""
        try:
            start_time = time.time()
            
            # Step 1: Query vector database
            vector_results = await self.query_vector_data(user_id, bot_id, query, top_k)
            
            if not vector_results:
                return {
                    "response": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "sources": [],
                    "total_chunks_found": 0,
                    "execution_time": time.time() - start_time
                }
            
            # Step 2: Generate OpenAI response
            ai_response = await self.generate_openai_response(
                query=query,
                context_chunks=vector_results,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Step 3: Prepare response
            sources = []
            for chunk in vector_results:
                sources.append({
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                    "relevance_score": chunk["score"],
                    "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                })
            
            execution_time = time.time() - start_time
            
            log.info(f"✅ Query completed in {execution_time:.2f}s")
            
            return {
                "response": ai_response,
                "sources": sources,
                "total_chunks_found": len(vector_results),
                "execution_time": execution_time
            }
            
        except Exception as e:
            log.error(f"Query and generation pipeline failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        try:
            start_time = time.time()
            
            # Test Upstash connection
            upstash_status = False
            try:
                # Try a simple embedding and query
                test_vector = self.embed_text("health check")
                test_result = self.index.query(
                    vector=test_vector,
                    top_k=1,
                    namespace="health_check"
                )
                upstash_status = True
            except Exception as e:
                log.warning(f"Upstash health check failed: {e}")
            
            # Test OpenAI connection
            openai_status = False
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
            except Exception as e:
                log.warning(f"OpenAI health check failed: {e}")
            
            overall_status = upstash_status and openai_status
            
            return {
                "overall_status": overall_status,
                "upstash_status": upstash_status,
                "openai_status": openai_status,
                "timestamp": datetime.now().isoformat(),
                "response_time": time.time() - start_time
            }
            
        except Exception as e:
            log.error(f"Health check failed: {e}")
            raise

    async def test_query(self) -> Dict[str, Any]:
        """Run a test query"""
        try:
            start_time = time.time()
            
            # Use a generic test query
            test_query = "What is the main topic of the documents?"
            
            # Try to find any available namespace for testing
            # This is a simple test - in production you might want to use specific test data
            test_response = await self.generate_openai_response(
                query=test_query,
                context_chunks=[{
                    "content": "This is a test document for API testing purposes.",
                    "source": "test_doc.txt",
                    "chunk_index": 0
                }],
                model="gpt-3.5-turbo",
                max_tokens=100
            )
            
            execution_time = time.time() - start_time
            
            return {
                "test_query": test_query,
                "response": test_response,
                "execution_time": execution_time,
                "status": "success"
            }
            
        except Exception as e:
            log.error(f"Test query failed: {e}")
            raise

    async def list_available_namespaces(self) -> List[str]:
        """List available namespaces (this is a simplified version)"""
        try:
            # Note: Upstash Vector doesn't have a direct way to list namespaces
            # This is a placeholder - you might need to maintain a registry
            # or use a different approach based on your setup
            
            log.info("Listing namespaces (placeholder implementation)")
            return ["user123_bot456", "test_namespace"]
            
        except Exception as e:
            log.error(f"Failed to list namespaces: {e}")
            raise

    async def get_namespace_stats(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """Get statistics for a namespace"""
        try:
            namespace = self.create_namespace(user_id, bot_id)
            
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
                metadata = result.get("metadata", {})
                if "source" in metadata:
                    sources.add(metadata["source"])
                if "content" in metadata:
                    total_content_length += len(metadata["content"])
            
            return {
                "namespace": namespace,
                "estimated_total_chunks": total_chunks,
                "unique_sources": len(sources),
                "sources": list(sources),
                "average_chunk_size": total_content_length // total_chunks if total_chunks > 0 else 0,
                "total_content_length": total_content_length
            }
            
        except Exception as e:
            log.error(f"Failed to get namespace stats: {e}")
            raise