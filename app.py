import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Convesation RAG with pdf upload and chat history")
st.write("upload pdf anc ahr with their content")

api_key=st.text_input("Enter your groq ASPI key" ,type="password")

#cheak if groq api key is provied 
if api_key:
    llm=ChatGroq(model="gemma2-9b-it",groq_api_key=api_key)

    session_id=st.text_input("Session ID",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store={}
    
    upload_file=st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)

    if upload_file:
        docuements=[]
        for uploaded_file in upload_file:
            temppdf=f'./temp.pdf'
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
            
            loader=PyPDFLoader(file_path=temppdf)
            doc=loader.load()
            docuements.extend(doc)

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(docuements)

        vectorstore=FAISS.from_documents(documents=splits,embedding=embedding)
        retriver=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_histroy"),
                    ("human", "{input}"),
                ]
            )

        history_aware_retirver=create_history_aware_retriever(llm, retriver, contextualize_q_prompt)

        system_prompt=(
            "You are an assistant for question-answering tasks ."
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "{context}"
        )
            
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_histroy"),
                ("human","{input}")
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm , qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retirver,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key='chat_histroy',
            output_messages_key='answer'
        )

        user_input=st.text_input("your QuestionD")
        if user_input:
            session_history=get_session_history(session_id)
            respones=conversation_rag_chain.invoke(
                {'input':user_input},
                config={
                    "configurable":{"session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.success(f"Assistant:{respones['answer']}")
            st.write("Chat History:",session_history.messages)
else:
    st.warning("pleace enter api key")