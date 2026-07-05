import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. NUEVAS IMPORTACIONES PARA LA MEMORIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Importaciones oficiales para la Arquitectura Corporativa de Azure
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Cargar variables de entorno del archivo seguro .env
load_dotenv()

st.set_page_config(page_title="Asistente RRHH - Azure AI Enterprise", page_icon="🔷")
st.title("🔷 Asistente Virtual de RRHH (Azure AI)")
st.markdown("Infraestructura corporativa RAG optimizada con servicios de Microsoft Azure.")

@st.cache_resource
def iniciar_rag_azure():
    # 1. Ingesta y segmentación del documento fuente
    loader = PyPDFLoader("manual_rrhh_novatech.pdf")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # 2. Pipeline de Embeddings (Indexación Vectorial)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # 3. Configuración del Motor de Inferencia (Azure OpenAI Service)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini-deployment")
    
    # Sistema Híbrido de Contingencia Inteligente
    if endpoint and api_key:
        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=deployment_name,
            api_version="2024-02-15-preview",
            temperature=0
        )
    else:
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    # 4. Diseño del Prompt (AHORA CON MEMORIA INYECTADA)
    system_prompt = (
        "Eres un asistente virtual de Recursos Humanos para la empresa NovaTech Solutions, "
        "desplegado mediante los servicios cognitivos de Microsoft Azure AI. "
        "Usa los siguientes fragmentos de contexto recuperado para responder a la pregunta en español. "
        "Si no sabes la respuesta o no está sustentada en el contexto, di exactamente: "
        "'Lo siento, no tengo esa información. Por favor contacta a RRHH directamente'. "
        "No inventes datos bajo ninguna circunstancia."
        "\n\n"
        "{context}"
    )
    
    # Añadimos MessagesPlaceholder para que el LLM reciba el contexto de la charla
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    # 5. Ensamblaje de la cadena RAG End-to-End
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Inicializar canalización optimizada de Azure
rag_chain = iniciar_rag_azure()

# Gestión del historial del Chat en memoria de sesión de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura y procesamiento del flujo de la conversación (Cambio de nombre de variable a user_question)
if user_question := st.chat_input("Ej: ¿Cuál es el canal oficial para reportar un conflicto de acoso?"):
    
    # Guardamos y mostramos la pregunta del usuario en la pantalla
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # 6. TRADUCCIÓN DEL HISTORIAL: Convertimos la memoria de Streamlit al formato de LangChain
    chat_history = []
    # Usamos [:-1] para pasarle todo el historial EXCEPTO la pregunta actual (que viaja en 'input')
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conocimiento indexada en Azure AI Search..."):
            # 7. INVOCACIÓN CON MEMORIA: Le pasamos la pregunta y todo el historial
            response = rag_chain.invoke({
                "input": user_question,
                "chat_history": chat_history
            })
            answer = response["answer"]
            st.markdown(answer)
    
    # Guardamos la respuesta del bot en la pantalla
    st.session_state.messages.append({"role": "assistant", "content": answer})
