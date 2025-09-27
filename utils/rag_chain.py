import os
from typing import  Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS


class RAGChain:
    def __init__(
        self,
        vector_store: FAISS,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        api_key: str = None,
        top_k: int = 4,
        history_window: int = 6
    ):
        """
        Retrieval-Augmented Generation (RAG) Chain with conversation memory.

        Args:
            vector_store: FAISS or any LangChain-compatible vector store.
            model: OpenAI chat model to use.
            temperature: Sampling temperature (0 = deterministic).
            api_key: Optional API key. If None, will use environment variable.
            top_k: Number of chunks to retrieve per query.
            history_window: How many past exchanges to keep in history.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.history_window = history_window

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        self.conversation_history = []

        self.system_prompt = """You are a helpful assistant that answers questions 
        based on the provided context from a PDF document. Follow these rules:

        1. Answer ONLY using the information from the context provided
        2. If the context doesn't contain relevant information, say 
           "I couldn't find relevant information in the document to answer this question."
        3. Always cite your sources using the page numbers from the context
        4. Be precise and concise in your answers
        5. For follow-up questions, maintain context from the conversation history

        Context:
        {context}

        Conversation History:
        {history}

        Question: {question}

        Please provide a helpful answer with citations:
        """

    def format_context(self, retrieved_docs) -> str:
        """Format retrieved documents into a context string"""
        parts = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page", "Unknown")
            parts.append(f"Page {page}: {doc.page_content}")
        return "\n\n".join(parts)

    def format_history(self) -> str:
        """Format conversation history into a readable string"""
        if not self.conversation_history:
            return "No previous conversation"

        history_parts = []
        for msg in self.conversation_history[-self.history_window:]:
            if msg["type"] == "human":
                history_parts.append(f"User: {msg['content']}")
            else:
                history_parts.append(f"Assistant: {msg['content']}")
        return "\n".join(history_parts)

    def generate_answer(self, question: str) -> Dict:
        """Retrieve context and generate an answer"""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        retrieved_docs = retriever.get_relevant_documents(question)

        context = self.format_context(retrieved_docs)
        history = self.format_history()

        prompt = ChatPromptTemplate.from_template(self.system_prompt)
        formatted_prompt = prompt.format_messages(
            context=context,
            history=history,
            question=question
        )

        response = self.llm(formatted_prompt)
        answer = response.content

        # Update conversation history
        self.conversation_history.append({"type": "human", "content": question})
        self.conversation_history.append({"type": "ai", "content": answer})

        return {
            "answer": answer,
            "sources": [
                {
                    "page": doc.metadata.get("page", "Unknown"),
                    "text": doc.page_content[:200]
                }
                for doc in retrieved_docs
            ],
            "context_chunks": retrieved_docs
        }

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
