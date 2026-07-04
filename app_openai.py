import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables de entorno (API Key)
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Asistente RRHH - NovaTech", page_icon="🤖")
st.title("🤖 Asistente Virtual de RRHH")
st.markdown("Consulta nuestras políticas, vacaciones y beneficios de NovaTech Solutions.")

# Función en caché para procesar el PDF una sola vez
@st.cache_resource
def iniciar_rag():
    # 1. Cargar el documento
    loader = PyPDFLoader("manual_rrhh_novatech.pdf")
    docs = loader.load()
    
    # 2. Dividir en chunks (fragmentos)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 3. Crear base de datos vectorial (Chroma)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # 4. Configurar el LLM y el Prompt estricto para evitar alucinaciones
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    system_prompt = (
        "Eres un asistente virtual de Recursos Humanos para la empresa NovaTech Solutions. "
        "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta. "
        "Si no sabes la respuesta o no está en el contexto, di exactamente: "
        "'Lo siento, no tengo esa información. Por favor contacta a RRHH directamente'. "
        "No inventes información bajo ninguna circunstancia. "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Crear la cadena
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Inicializar el motor RAG
rag_chain = iniciar_rag()

# Inicializar el historial de chat en Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar input del usuario
if prompt := st.chat_input("Ej: ¿Cuántos días de vacaciones tengo si llevo 2 años?"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en el manual..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
    
    # Guardar respuesta del bot
    st.session_state.messages.append({"role": "assistant", "content": answer})