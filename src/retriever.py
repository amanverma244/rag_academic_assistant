import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

class VectorStoreManager:
    def __init__(self, db_path="data/chroma_db", collection_name="academic_papers"):
        # Initialize local persistent Chroma client
        self.client = PersistentClient(path=db_path)
        
        # Load a powerful, lightweight open-source embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks):
        """Embeds and upserts document chunks into ChromaDB."""
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{chunk['metadata']['source']}_p{chunk['metadata']['page']}_{idx}"
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
            
        # Generate vector embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents).tolist()

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully added {len(ids)} chunks to the vector store.")

    def search(self, query: str, top_k=3):
        """Performs semantic similarity search on the vector database."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results

if __name__ == "__main__":
    # Test block using PDFIngestor from src.ingest
    from src.ingest import PDFIngestor
    
    sample_pdf = "data/raw_pdfs/sample.pdf"
    if os.path.exists(sample_pdf):
        ingestor = PDFIngestor(sample_pdf)
        pages = ingestor.extract_text_with_metadata()
        chunks = ingestor.chunk_documents(pages)
        
        vector_manager = VectorStoreManager()
        vector_manager.add_documents(chunks)
        
        query = "What is the main contribution of this paper?"
        search_results = vector_manager.search(query, top_k=2)
        print("\nSearch Results:")
        for doc, meta in zip(search_results['documents'][0], search_results['metadatas'][0]):
            print(f"- [Page {meta['page']}] {doc[:150]}...")