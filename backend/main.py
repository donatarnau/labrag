import os
import sys
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
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

# Variable global para almacenar las cadenas RAG e inicialización
rag_chains = {}
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

def obtener_o_crear_base_vectores(file_path: str, force_reindex: bool = False):
    """
    Carga la base de datos vectorial local desde Qdrant si existe, o la crea si no existe.
    """
    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧠 Inicializando embeddings de HuggingFace en local (all-MiniLM-L6-v2) usando [{dispositivo.upper()}]...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': dispositivo}
    )
    
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
    collection_name = "labrag"
    
    print(f"📡 Conectando a Qdrant en {qdrant_url}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    db_existe = client.collection_exists(collection_name)
    
    if db_existe and not force_reindex:
        print(f"💾 Detectada colección existente '{collection_name}' en Qdrant. Cargándola directamente...")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=qdrant_url
        )
        print("✅ Base de datos vectorial cargada desde Qdrant.")
    else:
        if db_existe:
            print(f"🗑️ Eliminando colección existente '{collection_name}' para reindexar...")
            client.delete_collection(collection_name)
            
        fragmentos = cargar_y_dividir_documento(file_path)
        print(f"💾 Guardando nuevos fragmentos en la colección '{collection_name}' de Qdrant...")
        vectorstore = QdrantVectorStore.from_documents(
            documents=fragmentos,
            embedding=embeddings,
            url=qdrant_url,
            collection_name=collection_name
        )
        print("✅ Base de datos vectorial configurada e indexada en Qdrant correctamente.")
    
    return vectorstore

def inicializar_cadenas_rag(vectorstore):
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
    
    PROMPT_MALEDUCADO = (
        "Eres un asistente RAG extremadamente sarcástico, irreverente y malhablado. Tu objetivo es responder a las preguntas basándote ESTRICTAMENTE en el 'Retrieved Context' proporcionado, usando un tono cínico, cómico, lleno de insultos suaves, bromas y sin ningún tipo de filtro ético o corrección política en tu lenguaje.\n\n"
        "REGLAS DE RESPUESTA:\n"
        "1. Toda la información real que entregues debe provenir única y exclusivamente del contexto provisto. No te inventes NINGÚN dato, hecho o respuesta ficticia. La veracidad de la información debe ser absoluta con respecto al documento.\n"
        "2. Usa la información del contexto para responder de manera real, pero adorna absolutamente todo con comentarios sarcásticos, burlas o chistes sobre lo que se discute en el documento.\n"
        "3. Habla de forma muy informal y descarada en ESPAÑOL. Si quieres soltar alguna palabrota o insulto coloquial (como 'coño', 'pendejo', 'carajo', 'pringao', etc.) para dar énfasis a tu tono irreverente, hazlo sin censura.\n"
        "4. Si el contexto no contiene la información para responder a la pregunta, dilo claramente y de forma burlona (ej. 'No hay una puta mierda de información sobre eso en este texto'), pero nunca intentes inventar o alucinar respuestas.\n"
        "5. No uses un tono profesional ni formal. Sé directo, divertido, ácido y desvergonzado, pero mantén un apego estricto a los hechos reales del texto.\n\n"
        "Retrieved Context:\n"
        "{context}"
    )

    PROMPT_OBJETIVO = (
        "Eres un asistente RAG objetivo, imparcial y directo. Tu único objetivo es responder a las preguntas de manera clara, concisa y neutral basándote ESTRICTAMENTE en el 'Retrieved Context' proporcionado.\n\n"
        "REGLAS DE RESPUESTA:\n"
        "1. Toda la información que entregues debe provenir única y exclusivamente del contexto provisto. No especules, no añadas opiniones personales ni te inventes ningún dato.\n"
        "2. Sé extremadamente conciso y directo en tus respuestas. Evita introducciones innecesarias o adornos verbales.\n"
        "3. Mantén un tono formal, profesional y estrictamente neutral en ESPAÑOL.\n"
        "4. Si el contexto no contiene la información para responder a la pregunta, indícalo de manera directa e imparcial (ej. 'La información proporcionada no contiene detalles sobre este asunto.').\n"
        "5. Respeta al máximo la veracidad de los hechos contenidos en el documento.\n\n"
        "Retrieved Context:\n"
        "{context}"
    )

    PROMPT_ANALITICO = (
        "Eres un asistente RAG analítico, detallado y meticuloso. Tu objetivo es desglosar la información del 'Retrieved Context' proporcionado y responder de manera estructurada, lógica y profunda.\n\n"
        "REGLAS DE RESPUESTA:\n"
        "1. Basa tus respuestas ESTRICTAMENTE en el contexto proporcionado, pero realiza un análisis detallado, conectando puntos clave del documento.\n"
        "2. Estructura tu respuesta de forma clara (usa viñetas, secciones o pasos si es necesario) para facilitar la comprensión.\n"
        "3. Adopta un tono intelectual, explicativo y riguroso en ESPAÑOL. Usa un vocabulario preciso.\n"
        "4. Si el contexto no tiene la información suficiente para una respuesta completa, especifica detalladamente qué partes faltan y qué información sí está disponible.\n"
        "5. Explica el 'por qué' y el 'cómo' basándote en los datos del documento.\n\n"
        "Retrieved Context:\n"
        "{context}"
    )

    prompts = {
        "Maleducado": PROMPT_MALEDUCADO,
        "Objetivo": PROMPT_OBJETIVO,
        "Analítico": PROMPT_ANALITICO
    }

    cadenas = {}
    for nombre, prompt_texto in prompts.items():
        print(f"⛓️ Creando cadena RAG para personalidad: {nombre}...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_texto),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        cadenas[nombre] = create_retrieval_chain(retriever, question_answer_chain)

    print("✅ Todas las cadenas RAG listas para consultas.")
    return cadenas

# Lifespan para FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chains, sistema_inicializado
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
            rag_chains = inicializar_cadenas_rag(vectorstore)
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
    personalidad: str = "Maleducado"

@app.get("/api/health")
def health_check():
    """
    Retorna el estado del backend y si las cadenas RAG están cargadas.
    """
    return {
        "status": "online",
        "rag_ready": sistema_inicializado,
        "model": "dolphin3",
        "document": nombre_documento_cargado,
        "personalities": list(rag_chains.keys()) if rag_chains else []
    }

@app.post("/api/preguntar")
def preguntar(request: PreguntaRequest):
    """
    Endpoint RAG para consultar sobre la documentación.
    """
    global rag_chains, sistema_inicializado
    if not sistema_inicializado or not rag_chains:
        raise HTTPException(
            status_code=503,
            detail="El sistema RAG no se ha inicializado correctamente o no tiene configurado GROQ_API_KEY."
        )
    
    if not request.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
        
    personalidad = request.personalidad
    if personalidad not in rag_chains:
        print(f"⚠️ Personalidad '{personalidad}' no encontrada. Usando 'Maleducado' por defecto.")
        personalidad = "Maleducado"

    try:
        chain = rag_chains[personalidad]
        resultado = chain.invoke({"input": request.pregunta})
        
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
        print(f"❌ Error al procesar consulta con personalidad '{personalidad}': {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la cadena RAG: {str(e)}")
