from pathlib import Path
import uuid
import json
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from upstash_vector import Index
from botforge.core.config import settings
from botforge.core.logger import log
import PyPDF2


class UpstashVectorUploader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        # Check PyMuPDF availability
    
        self.pymupdf_available = False
        print("⚠️ PyMuPDF not available. Using PyPDF2 only. Install with: pip install PyMuPDF")
        
        try:
            self.model: SentenceTransformer = SentenceTransformer("BAAI/bge-small-en-v1.5")
            print("✅ Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading sentence transformer: {e}")
            raise
            
        try:
            self.index: Index = Index(
                url=settings.upstash_url,
                token=settings.upstash_token
            )
            print("✅ Upstash Vector index initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Upstash Vector index: {e}")
            print(f"URL: {settings.upstash_url}")
            print(f"Token: {'*' * min(len(settings.upstash_token), 10) if settings.upstash_token else 'None'}")
            raise
            
        self.logger = log
        # Optimal chunk size based on research: ~250 tokens = ~1000 characters
        self.chunk_size = chunk_size  
        # Reduced overlap - too much overlap can hurt performance
        self.chunk_overlap = chunk_overlap
        
        print(f"📊 Chunking config: size={self.chunk_size}, overlap={self.chunk_overlap}")

    def get_files(self, user_id: str, bot_id: str) -> List[Path]:
        base_path = Path(f"{settings.upload_location}/{user_id}/{bot_id}")
        if not base_path.exists():
            self.logger.warning(f"Path does not exist: {base_path}")
            return []
        
        files = (list(base_path.glob("*.txt")) + 
                list(base_path.glob("*.json")) + 
                list(base_path.glob("*.pdf")))
        
        print(f"Found {len(files)} files in {base_path}")
        return files

    def extract_pdf_text_pymupdf(self, file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF (fitz) - more reliable for complex PDFs"""
        if not self.pymupdf_available:
            return ""
            
        try:
            import fitz
            doc = fitz.open(str(file_path))
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
            print(f"Using PyPDF2 to extract text from: {file_path.name}")
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                print(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text
                        print(f"Page {page_num + 1}: extracted {len(page_text)} characters")
                    except Exception as e:
                        print(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
                
                return text.strip()
        except Exception as e:
            self.logger.error(f"PyPDF2 failed for {file_path.name}: {e}")
            print(f"❌ PyPDF2 error: {e}")
            return ""

    def load_text(self, file_path: Path) -> str:
        """Load text from various file formats with better error handling"""
        try:
            print(f"Loading text from: {file_path.name}")
            
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8").strip()
                print(f"Loaded {len(text)} characters from TXT file")
                return text
                
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("text", "").strip()
                    print(f"Loaded {len(text)} characters from JSON file")
                    return text
                    
            elif file_path.suffix.lower() == ".pdf":
                print(f"Processing PDF: {file_path.name}")
                
                # Try PyMuPDF first (usually more reliable)
                text = self.extract_pdf_text_pymupdf(file_path) if self.pymupdf_available else ""
                if not text:
                    # Fallback to PyPDF2
                    print(f"Trying PyPDF2 as fallback for {file_path.name}")
                    text = self.extract_pdf_text_pypdf2(file_path)
                
                if not text:
                    self.logger.warning(f"No text extracted from PDF: {file_path.name}")
                    print(f"❌ Failed to extract text from PDF: {file_path.name}")
                else:
                    print(f"✅ Extracted {len(text)} characters from PDF: {file_path.name}")
                
                return text
        except Exception as e:
            self.logger.error(f"Error reading {file_path.name}: {e}")
        return ""

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Advanced text chunking with semantic-aware splitting.
        
        Based on RAG best practices:
        - ~250 tokens (1000 chars) is optimal starting point
        - Preserve sentence and paragraph boundaries
        - Maintain semantic coherence
        - Handle different document structures
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
            
        if not text or not text.strip():
            return []
            
        # Clean the text first
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace but preserve paragraphs
        text = text.strip()
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        # First, try to split by major document sections (headers, etc.)
        sections = self._split_by_sections(text)
        
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Further split large sections
                section_chunks = self._split_section_intelligently(section, chunk_size, chunk_overlap)
                chunks.extend(section_chunks)
        
        # Post-process chunks to ensure quality
        final_chunks = self._postprocess_chunks(chunks, chunk_size, chunk_overlap)
        
        print(f"Created {len(final_chunks)} semantic chunks from {len(text)} characters")
        
        # Log chunk size distribution for debugging
        sizes = [len(chunk) for chunk in final_chunks]
        if sizes:
            print(f"Chunk sizes - Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes)//len(sizes)}")
        
        return final_chunks

    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by major sections (headers, paragraphs, etc.)"""
        # Split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        
        # Look for headers (lines that are short and followed by content)
        sections = []
        current_section = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if this looks like a header (short line, often capitalized)
            if (len(para) < 100 and 
                (para.isupper() or para.istitle() or para.endswith(':')) and 
                current_section):
                # Start new section
                if current_section:
                    sections.append(current_section.strip())
                current_section = para + "\n\n"
            else:
                current_section += para + "\n\n"
        
        if current_section:
            sections.append(current_section.strip())
        
        return sections if sections else [text]

    def _split_section_intelligently(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a section using semantic boundaries"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        # Split by sentences first
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving meaning"""
        # Enhanced sentence splitting that handles abbreviations better
        sentence_endings = r'[.!?]+(?:\s|$)'
        potential_sentences = re.split(f'({sentence_endings})', text)
        
        sentences = []
        current = ""
        
        for i in range(0, len(potential_sentences), 2):
            sentence_part = potential_sentences[i] if i < len(potential_sentences) else ""
            ending_part = potential_sentences[i + 1] if i + 1 < len(potential_sentences) else ""
            
            current += sentence_part + ending_part
            
            # Check if this is a real sentence ending (not abbreviation)
            if ending_part and not self._is_abbreviation(sentence_part):
                sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if s.strip()]

    def _is_abbreviation(self, text: str) -> bool:
        """Check if the text before a period is likely an abbreviation"""
        if not text:
            return False
        
        words = text.strip().split()
        if not words:
            return False
            
        last_word = words[-1].lower()
        
        # Common abbreviations
        abbreviations = {
            'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'corp', 'co',
            'etc', 'vs', 'e.g', 'i.e', 'fig', 'vol', 'no', 'p', 'pp'
        }
        
        return (last_word in abbreviations or 
                (len(last_word) <= 3 and last_word.isalpha()))

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last part of text for overlap, preferring sentence boundaries"""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a sentence boundary within the overlap region
        overlap_start = len(text) - overlap_size
        sentence_end = -1
        
        for punct in ['.', '!', '?']:
            pos = text.rfind(punct, overlap_start)
            if pos > sentence_end and pos > overlap_start:
                sentence_end = pos
        
        if sentence_end > -1:
            return text[sentence_end + 1:].strip()
        else:
            # Fall back to word boundary
            space_pos = text.rfind(' ', overlap_start)
            if space_pos > overlap_start:
                return text[space_pos + 1:].strip()
            else:
                return text[-overlap_size:]

    def _postprocess_chunks(self, chunks: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Post-process chunks to optimize size and quality"""
        if not chunks:
            return []
        
        processed = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # If chunk is too small, try to merge with previous
            if (len(chunk) < chunk_size // 4 and processed and 
                len(processed[-1]) + len(chunk) <= chunk_size):
                processed[-1] += " " + chunk
            else:
                processed.append(chunk)
        
        return processed

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings with error handling"""
        try:
            if not text or not text.strip():
                raise ValueError("Empty text provided for embedding")
            
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def create_namespace(self, user_id: str, bot_id: str) -> str:
        """Create namespace from user_id and bot_id"""
        namespace = f"{user_id}_{bot_id}"
        print(f"Using namespace: {namespace}")
        return namespace

    def upsert_file(self, file_path: Path, user_id: str, bot_id: str) -> Dict[str, int]:
        """Upload a file by chunking it and inserting each chunk as a separate vector"""
        try:
            print(f"\n📁 Processing file: {file_path.name}")
            
            text = self.load_text(file_path)
            if not text:
                self.logger.warning(f"Skipping empty or unreadable file: {file_path.name}")
                return {"uploaded": 0, "skipped": 1}
            
            chunks = self.chunk_text(text)
            if not chunks:
                self.logger.warning(f"No chunks created from file: {file_path.name}")
                return {"uploaded": 0, "skipped": 1}
            
            namespace = self.create_namespace(user_id, bot_id)
            vectors_to_upsert = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    vector = self.embed_text(chunk)
                    if not vector:
                        print(f"⚠️ Empty vector for chunk {i}")
                        continue
                    
                    # Create unique ID (simpler format)
                    doc_id = f"{file_path.stem}_chunk_{i}_{uuid.uuid4().hex[:8]}"
                    
                    # Store FULL content in metadata
                    metadata = {
                        "source": file_path.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_length": len(chunk),
                        "original_file_length": len(text),
                        "user_id": user_id,
                        "bot_id": bot_id,
                        "namespace": namespace,
                        "content": chunk,  # FULL CONTENT HERE
                        "text": chunk,    # Also store as 'text' for compatibility
                    }
                    
                    vectors_to_upsert.append({
                        "id": doc_id,
                        "vector": vector,
                        "data": chunk
                    })
                    
                    print(f"✅ Prepared chunk {i+1}/{len(chunks)} (ID: {doc_id}, Length: {len(chunk)})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {i} from {file_path.name}: {e}")
                    continue
            
            # Batch upsert
            if vectors_to_upsert:
                try:
                    print(f"🚀 Upserting {len(vectors_to_upsert)} vectors to namespace: {namespace}")
                    
                    self.index.upsert(
                        vectors=vectors_to_upsert,
                        namespace=namespace
                    )
                    
                    uploaded_chunks = len(vectors_to_upsert)
                    print(f"✅ Successfully uploaded: {file_path.name} ({uploaded_chunks} chunks)")
                    self.logger.info(f"✅ Uploaded: {file_path.name} ({uploaded_chunks} chunks) to namespace: {namespace}")
                    
                    return {"uploaded": uploaded_chunks, "skipped": 0}
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed batch upsert for {file_path.name}: {e}")
                    print(f"❌ Upsert error: {e}")
                    return {"uploaded": 0, "skipped": len(vectors_to_upsert)}
            else:
                print(f"⚠️ No vectors to upsert for {file_path.name}")
                return {"uploaded": 0, "skipped": 1}
            
        except Exception as e:
            self.logger.error(f"❌ Failed to upload {file_path.name}: {e}")
            print(f"❌ File processing error: {e}")
            return {"uploaded": 0, "skipped": 1}

    def upload_user_bot_data(self, user_id: str, bot_id: str) -> Dict[str, Any]:
        """Upload all files for a specific user and bot combination"""
        print(f"\n🚀 Starting upload for user: {user_id}, bot: {bot_id}")
        
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
            print(f"\n📄 Processing file {files_processed + 1}/{len(files)}: {file.name}")
            result = self.upsert_file(file, user_id, bot_id)
            total_uploaded += result["uploaded"]
            total_skipped += result["skipped"]
            files_processed += 1

        print(f"\n📊 Upload Summary:")
        print(f"   Files processed: {files_processed}")
        print(f"   Chunks uploaded: {total_uploaded}")
        print(f"   Chunks skipped: {total_skipped}")
        print(f"   Namespace: {namespace}")
        
        self.logger.info(f"Upload complete for {namespace}: {files_processed} files, {total_uploaded} chunks uploaded, {total_skipped} chunks skipped")
        
        return {
            "files_processed": files_processed,
            "chunks_uploaded": total_uploaded, 
            "chunks_skipped": total_skipped,
            "namespace": namespace
        }

    def query_user_bot_data(self, query: str, user_id: str, bot_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query vectors from a specific user/bot namespace with proper display format"""
        try:
            namespace = self.create_namespace(user_id, bot_id)
            print(f"🔍 Querying namespace: {namespace} with query: '{query[:50]}...'")
            
            query_vector = self.embed_text(query)
            
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            print(f"📊 Found {len(results)} results")
            print("=" * 60)
            
            # Process and display results in your desired format
            processed_results = []
            for i, result in enumerate(results):
                # Extract the full content from metadata
                metadata = result.get("metadata", {})
                content = metadata.get("content", metadata.get("text", "No content found"))
                
                processed_result = {
                    "id": result.get("id"),
                    "score": result.get("score"),
                    "content": content,
                    "source": metadata.get("source", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "metadata": metadata
                }
                processed_results.append(processed_result)
                
                # Display in your desired format
                print(f"{result.get('id')}")
                print(f"Data:")
                print(content)
                print(f"Vector:")
                print(f"[ {', '.join(map(str, result.get('vector', [])[:6]))}...")
                print("=" * 60)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Query failed for namespace {namespace}: {e}")
            print(f"❌ Query error: {e}")
            return []

    def inspect_stored_data(self, user_id: str, bot_id: str, num_samples: int = 3) -> List[Dict[str, Any]]:
        """Inspect what's actually stored - formatted like you want"""
        try:
            namespace = self.create_namespace(user_id, bot_id)
            print(f"🔍 Inspecting stored data in namespace: {namespace}")
            
            # Create a dummy query to get some results
            dummy_query = "sample query"
            query_vector = self.embed_text(dummy_query)
            
            results = self.index.query(
                vector=query_vector,
                top_k=num_samples,
                namespace=namespace,
                include_metadata=True
            )
            
            print(f"📊 Inspecting {len(results)} sample records:")
            print("=" * 60)
            
            for i, result in enumerate(results):
                metadata = result.get('metadata', {})
                content = metadata.get('content', metadata.get('text', 'No content found'))
                
                # Display in your preferred format
                print(f"{result.get('id')}")
                print(f"Data:")
                print(content)
                print(f"Vector:")
                print(f"[ {', '.join(map(str, result.get('vector', [])[:6]))}...")
                print("=" * 60)
            
            return results
            
        except Exception as e:
            print(f"❌ Inspection failed: {e}")
            return []

    def debug_single_record(self, user_id: str, bot_id: str) -> None:
        """Debug a single record to see exactly what's stored"""
        try:
            namespace = self.create_namespace(user_id, bot_id)
            
            # Get one record
            dummy_query = "test"
            query_vector = self.embed_text(dummy_query)
            
            results = self.index.query(
                vector=query_vector,
                top_k=1,
                namespace=namespace,
                include_metadata=True
            )
            
            if results:
                result = results[0]
                print("🔍 DEBUG: Raw result structure:")
                print(f"Keys in result: {list(result.keys())}")
                
                metadata = result.get('metadata', {})
                print(f"Keys in metadata: {list(metadata.keys())}")
                
                # Check what text fields exist
                for key in ['content', 'text', 'data']:
                    if key in metadata:
                        print(f"Found text in '{key}': {len(metadata[key])} chars")
                        print(f"Preview: {metadata[key][:100]}...")
                
                # Show the desired format
                print("\n" + "="*50)
                print("FORMATTED OUTPUT:")
                print(f"{result.get('id')}")
                print("Data:")
                content = metadata.get('content', metadata.get('text', 'No content available'))
                print(content)
                print("Vector:")
                print(f"[ {', '.join(map(str, result.get('vector', [])[:6]))}...")
            else:
                print("No results found")
                
        except Exception as e:
            print(f"Debug failed: {e}")

    def delete_user_bot_data(self, user_id: str, bot_id: str) -> bool:
        """Delete all vectors for a specific user/bot namespace"""
        try:
            namespace = self.create_namespace(user_id, bot_id)
            print(f"🗑️ Deleting all data for namespace: {namespace}")
            
            # Note: Check Upstash Vector API documentation for correct deletion method
            self.index.delete(namespace=namespace, delete_all=True)
            self.logger.info(f"Deleted all data for namespace: {namespace}")
            print(f"✅ Successfully deleted namespace: {namespace}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete namespace {namespace}: {e}")
            print(f"❌ Deletion error: {e}")
            return False

    def test_connection(self) -> bool:
        """Test the connection to Upstash Vector"""
        try:
            # Try to get index info or perform a simple operation
            print("🔌 Testing Upstash Vector connection...")
            
            # Create a test vector
            test_vector = self.embed_text("test connection")
            test_id = f"test-{uuid.uuid4().hex[:8]}"
            
            # Try to upsert and then delete
            self.index.upsert(
                vectors=[{
                    "id": test_id,
                    "vector": test_vector,
                    "metadata": {"test": True}
                }],
                namespace="test"
            )
            
            # Clean up test vector
            self.index.delete(ids=[test_id], namespace="test")
            
            print("✅ Connection test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


# Usage example with better debugging:
if __name__ == "__main__":
    try:
        # Initialize with optimized parameters based on RAG research
        # 1000 chars ≈ 250 tokens (optimal for most embeddings)
        # 100 char overlap (10% overlap is usually sufficient)
        uploader = UpstashVectorUploader(chunk_size=1000, chunk_overlap=100)
        
        # Test connection first
        if not uploader.test_connection():
            print("❌ Connection test failed. Check your Upstash credentials.")
            exit(1)
        
        # First, debug what's currently stored
        print("\nDEBUGGING CURRENT DATA:")
        uploader.debug_single_record("user123", "bot456")
        
        # Inspect current data format
        print("\n" + "="*60)
        print("INSPECTING CURRENT DATA FORMAT:")
        uploader.inspect_stored_data("user123", "bot456", num_samples=2)
        
        # If you need to re-upload (uncomment these lines)
        # print("\n" + "="*60)
        # print("RE-UPLOADING DATA:")
        # uploader.delete_user_bot_data("user123", "bot456")  # Clear old data
        # result = uploader.upload_user_bot_data("user123", "bot456")  # Re-upload
        # print(f"\n📈 Upload Result: {result}")
        
        # Query the data
        print("\n" + "="*60)
        print("QUERYING DATA:")
        query_results = uploader.query_user_bot_data("PDF sample", "user123", "bot456", top_k=3)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        raise