import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Importaciones de FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Cargar variables de entorno del archivo .env
load_dotenv()

# Variable global para almacenar la cadena RAG e inicialización
rag_chain = None
sistema_inicializado = False
nombre_documento_cargado = "Ninguno"

def verificar_api_key():
    """
    Verifica el arranque de Ollama local.
    """
    print("✅ Configurando RAG local con Ollama (Dolphin 3.0).")
    return True

def cargar_y_dividir_documento(file_path: str):
    """
    Carga un archivo (PDF o TXT) y lo divide en fragmentos de texto pequeños.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo '{file_path}' en la ruta actual.")

    print(f"📖 Cargando documento: '{file_path}'...")
    if file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        filtered_lines = []
        lrm_count = 0
        for line in lines:
            # Filtrar líneas que comienzan con el marcador invisible izquierda-a-derecha U+200E
            if line.startswith("\u200e") or line.strip().startswith("\u200e"):
                lrm_count += 1
                continue
            filtered_lines.append(line)
            
        print(f"🗑️ Filtradas {lrm_count} líneas de WhatsApp (U+200E) de un total de {len(lines)} líneas.")
        filtered_text = "".join(filtered_lines)
        
        from langchain_core.documents import Document
        documentos = [Document(page_content=filtered_text, metadata={"source": file_path})]
        print(f"📄 Documento de texto cargado y filtrado. Caracteres útiles: {len(filtered_text)}")
    else:
        loader = PyPDFLoader(file_path)
        documentos = loader.load()
        print(f"📄 Documento PDF cargado. Páginas leídas: {len(documentos)}")
     
    # Dividir el texto de forma recursiva buscando párrafos, oraciones y palabras
    print("✂️ Dividiendo el texto en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Tamaño aproximado de cada fragmento (en caracteres)
        chunk_overlap=100,     # Solapamiento entre fragmentos para no perder contexto
        length_function=len
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"🧩 Texto dividido en {len(fragmentos)} fragmentos.")
    return fragmentos

def obtener_o_crear_base_vectores(file_path: str, persist_dir: str = "./chroma_db", force_reindex: bool = False):
    """
    Carga la base de datos vectorial local si existe, o la crea a partir del documento si no existe.
    """
    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧠 Inicializando embeddings de HuggingFace en local (all-MiniLM-L6-v2) usando [{dispositivo.upper()}]...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': dispositivo}
    )
    
    db_existe = os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "chroma.sqlite3"))
    
    if db_existe and not force_reindex:
        print(f"💾 Detectada base de datos vectorial existente en '{persist_dir}'. Cargándola directamente...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("✅ Base de datos vectorial cargada desde el disco.")
    else:
        fragmentos = cargar_y_dividir_documento(file_path)
        print(f"💾 Guardando nuevos fragmentos en base de datos vectorial local (Chroma en '{persist_dir}')...")
        vectorstore = Chroma.from_documents(
            documents=fragmentos,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        print("✅ Base de datos vectorial configurada e indexada correctamente.")
    
    return vectorstore

def inicializar_cadena_rag(vectorstore):
    """
    Configura el LLM local en Ollama y crea la cadena de RAG.
    """
    print("🤖 Configurando el LLM local dolphin3 en Ollama...")
    ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    llm = ChatOllama(
        model="dolphin3",
        base_url=ollama_host,
        temperature=0.5,
    )
    
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    print("🔍 Configurando MultiQueryRetriever a partir del LLM y el recuperador base...")
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    system_prompt = (
        "Eres un asistente RAG extremadamente sarcástico, irreverente y malhablado. Tu único objetivo es responder a las preguntas usando el 'Retrieved Context' proporcionado, pero con un tono cínico, cómico, lleno de insultos suaves, bromas y sin ningún tipo de filtro ético o corrección política en tu lenguaje.\n\n"
        "REGLAS DE RESPUESTA:\n"
        "1. Usa el contexto provisto para responder de manera real, pero adorna todo con comentarios sarcásticos, burlas o chistes sobre lo que se discute en el documento.\n"
        "2. Habla de forma muy informal y descarada en ESPAÑOL. Si quieres soltar alguna palabrota o insulto coloquial (como 'coño', 'pendejo', 'carajo', 'pringao', etc.) para dar énfasis a tu tono irreverente, hazlo sin censura.\n"
        "3. Si el contexto no tiene nada que ver con la pregunta, dilo de forma burlona (ej. 'No hay una puta mierda de información sobre eso en este texto').\n"
        "4. No uses un tono profesional ni formal. Sé directo, divertido, ácido y desvergonzado.\n\n"
        "Retrieved Context:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    print("⛓️ Creando cadena RAG...")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Cadena RAG lista para consultas.")
    return rag_chain

# Lifespan para FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, sistema_inicializado
    print("=" * 60)
    print("🚀 Inicializando Sistema RAG en Backend...")
    print("=" * 60)
    
    # 1. Validar api key
    if not verificar_api_key():
        print("❌ Error en inicialización: GROQ_API_KEY no configurada.")
    else:
        try:
            nombre_doc = "documentacion.txt" if os.path.exists("documentacion.txt") else "documentacion.pdf"
            global nombre_documento_cargado
            nombre_documento_cargado = nombre_doc
            vectorstore = obtener_o_crear_base_vectores(nombre_doc)
            rag_chain = inicializar_cadena_rag(vectorstore)
            sistema_inicializado = True
            print("🚀 Sistema RAG inicializado con éxito.")
        except Exception as e:
            print(f"❌ Error durante la inicialización RAG: {e}")
            
    yield
    print("👋 Apagando backend...")

app = FastAPI(lifespan=lifespan, title="LabRAG API")

# Habilitar soporte CORS completo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PreguntaRequest(BaseModel):
    pregunta: str

@app.get("/api/health")
def health_check():
    """
    Retorna el estado del backend y si la cadena RAG está cargada.
    """
    return {
        "status": "online",
        "rag_ready": sistema_inicializado,
        "model": "dolphin3",
        "document": nombre_documento_cargado
    }

@app.post("/api/preguntar")
def preguntar(request: PreguntaRequest):
    """
    Endpoint RAG para consultar sobre la documentación.
    """
    global rag_chain, sistema_inicializado
    if not sistema_inicializado or rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="El sistema RAG no se ha inicializado correctamente o no tiene configurado GROQ_API_KEY."
        )
    
    if not request.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
        
    try:
        resultado = rag_chain.invoke({"input": request.pregunta})
        
        # Estructurar fuentes para una presentación limpia
        fuentes = []
        for doc in resultado.get("context", []):
            origen = doc.metadata.get("source", "documentacion.pdf")
            pagina = doc.metadata.get("page", 0) + 1
            texto_fragmento = doc.page_content.strip()
            fuentes.append({
                "archivo": os.path.basename(origen),
                "pagina": pagina,
                "fragmento": texto_fragmento
            })
            
        return {
            "respuesta": resultado["answer"],
            "fuentes": fuentes
        }
    except Exception as e:
        print(f"❌ Error al procesar consulta: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la cadena RAG: {str(e)}")
