from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from src.retriever import VectorStoreManager

class RAGPipeline:
    def __init__(self):
        self.client = genai.Client()
        self.vectore_manager = VectorStoreManager()
        
        self.model_name = "gemini-3.6-flash"
        
    def answer_query(self,query: str, top_k=3):
        search_results = self.vectore_manager.search(query, top_k=top_k) 
        chunks = search_results['documents'][0]
        metadatas = search_results['metadatas'][0]
        
        context_blocks = []
        for i , (chunk,meta) in enumerate(zip(chunks,metadatas)):
            source_info = f"[Source: {meta['source']}, Page : {meta['page']}]"
            context_blocks.append(f"Context Snippet {i+1} {source_info}:\n{chunk}")
            
        joined_context = "\n\n".join (context_blocks)
        
        prompt = f"""you are an expert AI academic research assistant. Answer the user's query accurately using ONLY the provided context snippets below.
        IF the answer cannot be found in the context, state clearly that you do not know based on the provided papers.
        Always cite your sources using the format ([Filename], Page X) when stating facts from the text.
        
        ---
        CONTEXT:
        {joined_context}
        ---
        
        USER QUERY: {query}
        """  
        response = self.client.models.generate_content(
            model = self.model_name,
            contents = prompt,
        )  
        
        return {
            "answer": response.text,
            "sources": metadatas
        }
        
if __name__ == "__main__":
    rag = RAGPipeline()
    query = "what is the main contribution of this paper?"
    
    result = rag.answer_query(query)
    print("--- GENERATED ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES USED ---")
    for meta in result["sources"]:
        print(f"- {meta['source']} (Page {meta['page']})")        
               