
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from utils.pdf_processor import PDFProcessor
from utils.rag_chain import RAGChain

def main():
    st.set_page_config(page_title="Chat with your PDF", layout="wide")
    st.title("Chat with your PDF")
    if "show_help" not in st.session_state:
        st.session_state.show_help = True
    if st.session_state.show_help:
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. Upload a PDF file  
            2. Click **Process PDF**  
            3. Ask questions about the content  
            4. View citations from the document  
            5. Ask follow-up questions  
            """)
    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- UI Components ---
    with st.sidebar:
        st.header("Upload your PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

        if st.button("Process PDF", disabled=(pdf_file is None)):
            if pdf_file is not None:
                with st.spinner("Processing PDF..."):
                    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
                    st.session_state.vector_store = processor.process_pdf(pdf_file)
                st.success("PDF processed successfully! You can now ask questions.")
                st.session_state.show_help = False
            else:
                st.error("Please upload a PDF file first.")
                st.markdown("---")
       
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)        
        st.markdown("### Current Model")
        st.markdown("**Embeddings:** text-embedding-3-small")
        st.markdown("**LLM:** gpt-3.5-turbo")
        st.markdown("**Chunk Size:** 1000 tokens")
        st.markdown("**Retrieval:** Top 4 chunks")

    # --- Chat Interface ---
    if st.session_state.vector_store is None:
        st.info("Please upload and process a PDF to begin the chat.")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    # User input
    user_query = st.chat_input("Ask a question about your document...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                st.session_state.rag_chain = RAGChain(st.session_state.vector_store)
                
                # Invoke the chain with chat history for conversational context
                result = st.session_state.rag_chain.generate_answer(user_query)
                st.markdown(result["answer"])

                # Sources in expandable section
                with st.expander("View Sources and Citations"):
                    st.markdown("**Relevant passages from the document:**")
                    for source in result["sources"]:
                        st.markdown(f"**Page {source['page']}:**")
                        st.markdown(f"*{source['text']}...*")
                        st.markdown("---")

                # Add assistant reply to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
        
        st.session_state.chat_history.append(AIMessage(result["answer"]))


if __name__ == "__main__":
    main()