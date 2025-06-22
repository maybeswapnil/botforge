from pathlib import Path
import uuid
import json
import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from botforge.core.config import settings
from botforge.core.logger import log
import PyPDF2


class UpstashVectorUploader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model: SentenceTransformer = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.index: Index = Index(
            url=settings.upstash_url,
            token=settings.upstash_token
        )
        self.logger = log
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_files(self, user_id: str, bot_id: str) -> List[Path]:
        base_path = Path(f"{settings.upload_location}/{user_id}/{bot_id}")
        if not base_path.exists():
            self.logger.warning(f"Path does not exist: {base_path}")
            return []
        return (list(base_path.glob("*.txt")) + 
                list(base_path.glob("*.json")) + 
                list(base_path.glob("*.pdf")))

    def extract_pdf_text_pymupdf(self, file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF (fitz) - more reliable for complex PDFs"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            self.logger.error(f"PyMuPDF failed for {file_path.name}: {e}")
            return ""

    def extract_pdf_text_pypdf2(self, file_path: Path) -> str:
        """Extract text from PDF using PyPDF2 - fallback method"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text.strip()
        except Exception as e:
            self.logger.error(f"PyPDF2 failed for {file_path.name}: {e}")
            return ""

    def load_text(self, file_path: Path) -> str:
        try:
            if file_path.suffix == ".txt":
                return file_path.read_text(encoding="utf-8").strip()
            elif file_path.suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("text", "").strip()
            elif file_path.suffix == ".pdf":
                # Try PyMuPDF first (usually more reliable)
                text = self.extract_pdf_text_pymupdf(file_path)
                if not text:
                    # Fallback to PyPDF2
                    self.logger.info(f"Trying PyPDF2 as fallback for {file_path.name}")
                    text = self.extract_pdf_text_pypdf2(file_path)
                
                if not text:
                    self.logger.warning(f"No text extracted from PDF: {file_path.name}")
                else:
                    self.logger.info(f"Extracted {len(text)} characters from {file_path.name}")
                
                return text
        except Exception as e:
            self.logger.error(f"Error reading {file_path.name}: {e}")
        return ""

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks of specified size.
        
        Args:
            text: The text to chunk
            chunk_size: Maximum characters per chunk (defaults to self.chunk_size)
            chunk_overlap: Number of characters to overlap between chunks (defaults to self.chunk_overlap)
        
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
            
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Define the end of the current chunk
            end = start + chunk_size
            
            # If this is not the last chunk, try to find a good breaking point
            if end < len(text):
                # Look for sentence boundaries near the end
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                # If no sentence boundary found, look for paragraph breaks
                if sentence_end == -1 or sentence_end < start + chunk_size // 2:
                    paragraph_end = text.rfind('\n\n', start, end)
                    if paragraph_end != -1 and paragraph_end > start + chunk_size // 2:
                        sentence_end = paragraph_end
                
                # If no good breaking point found, look for any whitespace
                if sentence_end == -1 or sentence_end < start + chunk_size // 2:
                    space_end = text.rfind(' ', start, end)
                    if space_end != -1 and space_end > start + chunk_size // 2:
                        sentence_end = space_end
                
                # Use the breaking point if found, otherwise use the hard limit
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk (with overlap)
            if end >= len(text):
                break
            start = max(start + 1, end - chunk_overlap)
        
        return chunks

    def embed_text(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def create_namespace(self, user_id: str, bot_id: str) -> str:
        """Create namespace from user_id and bot_id"""
        return f"{user_id}_{bot_id}"

    def upsert_file(self, file_path: Path, user_id: str, bot_id: str) -> Dict[str, int]:
        """
        Upload a file by chunking it and inserting each chunk as a separate vector.
        
        Returns:
            Dict with counts of uploaded and skipped chunks
        """
        try:
            text = self.load_text(file_path)
            if not text:
                self.logger.warning(f"Skipping empty or unreadable file: {file_path.name}")
                return {"uploaded": 0, "skipped": 1}
            
            # Chunk the text
            chunks = self.chunk_text(text)
            namespace = self.create_namespace(user_id, bot_id)
            
            uploaded_chunks = 0
            skipped_chunks = 0
            
            # Process each chunk
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                try:
                    vector = self.embed_text(chunk)
                    doc_id = f"{file_path.stem}-chunk-{i}-{uuid.uuid4().hex[:6]}"
                    metadata = {
                        "source": file_path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_length": len(chunk),
                        "original_file_length": len(text),
                        "user_id": user_id,
                        "bot_id": bot_id,
                        "namespace": namespace
                    }
                    
                    vectors_to_upsert.append({
                        "id": doc_id,
                        "vector": vector,
                        "metadata": metadata
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {i} from {file_path.name}: {e}")
                    skipped_chunks += 1
            
            # Batch upsert all vectors for this file
            if vectors_to_upsert:
                try:
                    self.index.upsert(
                        vectors=vectors_to_upsert,
                        namespace=namespace
                    )
                    uploaded_chunks = len(vectors_to_upsert)
                    self.logger.info(f"✅ Uploaded: {file_path.name} ({uploaded_chunks} chunks) to namespace: {namespace}")
                except Exception as e:
                    self.logger.error(f"❌ Failed batch upsert for {file_path.name}: {e}")
                    skipped_chunks = len(vectors_to_upsert)
                    uploaded_chunks = 0
            
            return {"uploaded": uploaded_chunks, "skipped": skipped_chunks}
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload {file_path.name}: {e}")
            return {"uploaded": 0, "skipped": 1}

    def upload_user_bot_data(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """
        Upload all files for a specific user and bot combination.
        
        Returns:
            Dict with upload statistics
        """
        files = self.get_files(user_id, bot_id)
        if not files:
            self.logger.warning(f"No data files found for {user_id}/{bot_id}")
            return {
                "files_processed": 0,
                "chunks_uploaded": 0, 
                "chunks_skipped": 0,
                "namespace": self.create_namespace(user_id, bot_id)
            }

        total_uploaded = 0
        total_skipped = 0
        files_processed = 0
        
        namespace = self.create_namespace(user_id, bot_id)
        self.logger.info(f"Starting upload for namespace: {namespace}")
        
        for file in files:
            result = self.upsert_file(file, user_id, bot_id)
            total_uploaded += result["uploaded"]
            total_skipped += result["skipped"]
            files_processed += 1

        self.logger.info(f"Upload complete for {namespace}: {files_processed} files, {total_uploaded} chunks uploaded, {total_skipped} chunks skipped")
        
        return {
            "files_processed": files_processed,
            "chunks_uploaded": total_uploaded, 
            "chunks_skipped": total_skipped,
            "namespace": namespace
        }

    def query_user_bot_data(self, query: str, user_id: str, bot_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query vectors from a specific user/bot namespace.
        
        Args:
            query: The search query
            user_id: User identifier
            bot_id: Bot identifier
            top_k: Number of results to return
            
        Returns:
            List of matching results with metadata
        """
        try:
            namespace = self.create_namespace(user_id, bot_id)
            query_vector = self.embed_text(query)
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Query failed for namespace {namespace}: {e}")
            return []

    def delete_user_bot_data(self, user_id: str, bot_id: str) -> bool:
        """
        Delete all vectors for a specific user/bot namespace.
        
        Args:
            user_id: User identifier
            bot_id: Bot identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            namespace = self.create_namespace(user_id, bot_id)
            # Note: This assumes Upstash Vector supports namespace deletion
            # You might need to implement this differently based on the API
            self.index.delete(namespace=namespace, delete_all=True)
            self.logger.info(f"Deleted all data for namespace: {namespace}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete namespace {namespace}: {e}")
            return False


# Usage example:
if __name__ == "__main__":
    uploader = UpstashVectorUploader(chunk_size=800, chunk_overlap=150)
    
    # Upload data for a specific user and bot
    result = uploader.upload_user_bot_data("user123", "bot456")
    print(f"Upload result: {result}")
    
    # Query the uploaded data
    query_results = uploader.query_user_bot_data("search query", "user123", "bot456", top_k=3)
    print(f"Query results: {query_results}")