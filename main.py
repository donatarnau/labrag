import os
import sys

# Configurar codificación UTF-8 en consola para evitar errores con Emojis en Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# Cargar variables de entorno del archivo .env
load_dotenv()

def verificar_api_key():
    """
    Verifica que la clave de API de Groq esté configurada en las variables de entorno.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or "tu_clave_api" in api_key:
        print("\n❌ ERROR: No se ha configurado la variable de entorno GROQ_API_KEY.")
        print("Por favor, crea un archivo '.env' basado en '.env.example' e introduce tu clave real de Groq.")
        print("Puedes obtener una gratis en: https://console.groq.com/\n")
        sys.exit(1)
    print("✅ GROQ_API_KEY detectada correctamente.")

def cargar_y_dividir_pdf(pdf_path: str):
    """
    Carga un archivo PDF y lo divide en fragmentos de texto pequeños.
    """
    if not os.path.exists(pdf_path):
        print(f"\n❌ ERROR: No se encontró el archivo PDF '{pdf_path}' en la ruta actual.")
        print("Por favor, asegúrate de que el documento esté en la misma carpeta que este script.\n")
        sys.exit(1)

    print(f"📖 Cargando documento: '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()
    print(f"📄 Documento cargado. Páginas leídas: {len(documentos)}")
     
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

def obtener_o_crear_base_vectores(pdf_path: str, persist_dir: str = "./chroma_db", force_reindex: bool = False):
    """
    Carga la base de datos vectorial local si existe, o la crea a partir del PDF si no existe.
    """
    # Detectar de forma automática si hay una GPU NVIDIA disponible con CUDA
    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧠 Inicializando embeddings de HuggingFace en local (all-MiniLM-L6-v2) usando [{dispositivo.upper()}]...")
    # Usamos HuggingFaceEmbeddings con un modelo ligero y eficiente
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': dispositivo}
    )
    
    # Si la base de datos ya existe y no forzamos reindexación, la cargamos directamente de disco
    if os.path.exists(persist_dir) and not force_reindex:
        print(f"💾 Detectada base de datos vectorial existente en '{persist_dir}'. Cargándola directamente...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("✅ Base de datos vectorial cargada desde el disco.")
    else:
        # Si no existe o se fuerza reindexación, procesamos el PDF de nuevo
        fragmentos = cargar_y_dividir_pdf(pdf_path)
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
    Configura el LLM en Groq y crea la cadena de RAG (Retrieval-Augmented Generation).
    """
    print("🤖 Configurando el LLM llama-3.1-8b-instant en Groq...")
    # Instanciamos el modelo exacto solicitado usando langchain-groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,  # Temperatura baja para respuestas basadas estrictamente en hechos
    )
    
    # Configuramos el recuperador base (Base Retriever)
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    # Creamos el MultiQueryRetriever usando el LLM para generar múltiples preguntas/búsquedas alternativas
    print("🔍 Configurando MultiQueryRetriever a partir del LLM y el recuperador base...")
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    # Definimos la plantilla de prompt para guiar al modelo
    system_prompt = (
    "You are an expert technical analyst. Your task is to answer questions based strictly on the provided 'Retrieved Context'.\n\n"
    "RESPONSE GUIDELINES:\n"
    "1. ALWAYS provide an answer if the context contains relevant or related information. Synthesize technical details, features, or flows into a clear response.\n"
    "2. DO NOT be overly restrictive. If the user asks about a general concept (like 'navigation') and the context describes specific components (like 'BottomNav', 'TopAppBar', or 'menus'), connect these points to answer the question.\n"
    "3. Use the context to 'deduce' logically. For example, if the document mentions design justifications, use them to explain the 'purpose'[cite: 13].\n"
    "4. ONLY if the context is absolutely irrelevant to the query, state: 'No se encuentra información específica sobre este punto en el documento.'\n"
    "5. IMPORTANT: Write your final response in SPANISH. Maintain a professional and formal tone.\n"
    "6. Use bullet points for complex explanations to improve readability.\n\n"
    "Retrieved Context:\n"
    "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    print("⛓️ Creando cadena RAG...")
    # Crea la cadena que toma los documentos recuperados y los formatea dentro del prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Crea la cadena final que primero realiza la búsqueda y luego pasa los resultados al LLM
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("✅ Cadena RAG lista para consultas.")
    return rag_chain

def ejecutar_consulta(rag_chain, pregunta: str):
    """
    Ejecuta una consulta sobre la cadena RAG y muestra la respuesta y las fuentes.
    """
    print(f"\n❓ Pregunta: '{pregunta}'")
    print("🔍 Buscando en el documento y generando respuesta...")
    
    # Invocar el pipeline RAG
    resultado = rag_chain.invoke({"input": pregunta})
    
    print("\n💡 Respuesta del Sistema:")
    print("-" * 50)
    print(resultado["answer"])
    print("-" * 50)
    
    # Opcional: Mostrar de dónde se obtuvo la información
    print("\n📄 Fuentes de Contexto Utilizadas:")
    for i, doc in enumerate(resultado["context"], 1):
        origen = doc.metadata.get("source", "Desconocido")
        pagina = doc.metadata.get("page", 0) + 1  # Las páginas en PyPDFLoader son 0-indexed
        print(f"  [{i}] Archivo: {os.path.basename(origen)} (Pág. {pagina})")
        # Mostrar los primeros 100 caracteres del fragmento para referencia
        texto_resumido = doc.page_content.strip().replace('\n', ' ')[:120]
        print(f"      Fragmento: \"{texto_resumido}...\"")
    print("-" * 50)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Iniciando Sistema RAG con Groq, HuggingFace y ChromaDB")
    print("=" * 60)
    
    # 1. Validar configuraciones
    verificar_api_key()
    
    # 2. Definir ruta del documento (usaremos el PDF provisto en tu carpeta)
    nombre_pdf = "documentacion.pdf"
    
    # 3. Flujo RAG modular: Obtener o crear la base de datos vectorial
    # La primera vez procesará el PDF; las siguientes se cargará en menos de un segundo
    vectorstore = obtener_o_crear_base_vectores(nombre_pdf)
    rag_chain = inicializar_cadena_rag(vectorstore)
    
    # 4. Bucle interactivo de consultas en consola
    print("\n" + "=" * 60)
    print("💬 ¡Sistema RAG listo! Escribe tus preguntas sobre el documento.")
    print("Escribe 'salir' o 'exit' para terminar la conversación.")
    print("=" * 60)
    
    while True:
        try:
            pregunta = input("\n👉 Pregunta: ")
            if pregunta.strip().lower() in ["salir", "exit"]:
                print("\n👋 ¡Hasta luego! ¡Gracias por usar el sistema RAG!")
                break
            if not pregunta.strip():
                continue
                
            ejecutar_consulta(rag_chain, pregunta.strip())
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
