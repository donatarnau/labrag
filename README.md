# LabRAG: Sistema RAG Local con Docker (FastAPI + Vue 3 + Ollama + Qdrant)

Este proyecto proporciona una arquitectura completa, profesional y dockerizada para implementar un sistema **RAG (Retrieval-Augmented Generation)** 100% local. Permite realizar consultas interactivas sobre documentos (`.pdf` o `.txt`) mediante una interfaz web moderna.

El backend cuenta con una personalidad muy particular: está configurado por defecto como un **asistente RAG extremadamente sarcástico, irreverente y humorístico** (usando el modelo `dolphin3` en Ollama).

---

## 🛠️ Arquitectura y Tecnologías

El sistema está orquestado con **Docker Compose** en cuatro contenedores principales:

1. **Frontend (Vue 3 + Vite)**: 
   - Una interfaz web moderna, responsiva, con tema oscuro premium y visualización interactiva de las fuentes de información de cada respuesta.
   - Servido mediante Nginx en el puerto `80`.
2. **Backend (FastAPI + LangChain)**:
   - API modular en Python.
   - **Embeddings locales**: HuggingFace (`all-MiniLM-L6-v2`) que se ejecutan localmente (con aceleración de CPU optimizada).
   - **MultiQueryRetriever**: Genera variaciones de la consulta para mejorar la tasa de aciertos y la relevancia de los fragmentos recuperados.
3. **Qdrant**:
   - Base de datos vectorial de alto rendimiento que almacena y recupera los fragmentos indexados.
   - Persistencia local mediante volumen mapeado en el puerto `6333`.
4. **Ollama**:
   - Motor de LLM local configurado por defecto para ejecutar el modelo `dolphin3`.
   - Soporte nativo para aceleración por GPU NVIDIA si está disponible en el host.

---

## 📂 Estructura del Proyecto

```text
labrag/
├── backend/
│   ├── main.py                  # API principal y lógica de LangChain RAG
│   ├── requirements.txt         # Librerías de Python (FastAPI, PyTorch, LangChain...)
│   ├── Dockerfile               # Construcción optimizada (multietapa, cache HuggingFace offline)
│   └── documentacion.txt        # Documento de texto por defecto (puedes usar también .pdf)
├── frontend/
│   ├── src/                     # Código fuente de la interfaz en Vue 3
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile               # Construcción multietapa con Nginx
├── docker-compose.yml           # Definición de servicios, volúmenes y GPU
├── qdrant_data/                 # [Ignorado en Git] Persistencia de vectores de Qdrant
├── ollama_data/                 # [Ignorado en Git] Modelos locales de Ollama
├── .gitignore                   # Configuración de archivos excluidos de Git
└── README.md                    # Esta guía
```

---

## 🚀 Guía de Configuración Rápida

### Requisitos Previos

- **Docker** y **Docker Compose** instalados.
- *(Opcional)* **NVIDIA Container Toolkit** para habilitar la aceleración por GPU.

### Paso 1: Preparar el Documento
Coloca tu documento de referencia dentro de la carpeta `backend/`. El sistema buscará:
- `backend/documentacion.txt` (si existe, filtrando marcadores de chat como los de WhatsApp).
- `backend/documentacion.pdf` (si no existe el archivo de texto).

*Nota: Asegúrate de que el archivo se llame exactamente `documentacion.txt` o `documentacion.pdf`.*

### Paso 2: Levantar el Entorno con Docker Compose
Desde la raíz del proyecto, ejecuta el siguiente comando para compilar e iniciar los servicios en segundo plano:

```bash
docker compose up --build -d
```

Este comando:
1. Compilará la interfaz de Vue y la montará en un servidor Nginx expuesto en el puerto `80`.
2. Descargará y compilará la imagen del backend de FastAPI, pre-descargando el modelo de embeddings local para que funcione de forma 100% offline.
3. Levantará la base de datos vectorial Qdrant.
4. Iniciará el contenedor de Ollama.

### Paso 3: Descargar el Modelo en Ollama
Dado que Ollama se ejecuta localmente y no requiere de APIs externas, debes descargar el modelo `dolphin3` ejecutando el siguiente comando:

```bash
docker compose exec ollama ollama run dolphin3
```

*(Una vez que comience a descargar el modelo, puedes cerrar el proceso interactivo con `Ctrl + C`. El modelo seguirá estando disponible en el contenedor de Ollama gracias al volumen persistente `ollama_data`).*

---

## 💻 Direcciones de Acceso

Una vez levantados los servicios y con el modelo descargado en Ollama, puedes acceder a:

* **Panel Frontend (Interfaz Web)**: [http://localhost](http://localhost) (Puerto 80)
* **Documentación de la API (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Consola de Qdrant (Dashboard)**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
* **Endpoint de Salud del Backend**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 🧠 ¿Cómo Funciona la lógica del RAG?

1. **Lectura y Fragmentación**: El backend procesa el documento y lo fragmenta en partes de 500 caracteres con 100 de solapamiento semántico.
2. **Embeddings Locales Offline**: Genera vectores utilizando `all-MiniLM-L6-v2` ejecutándose de manera local en el contenedor.
3. **Búsqueda Vectorial**: Expande y optimiza la búsqueda de fragmentos relevantes en la base de datos de Qdrant mediante `MultiQueryRetriever`.
4. **Generación con Personalidad**: El contexto recuperado es inyectado en un prompt diseñado específicamente para que el LLM responda con un rol irreverente y sarcástico.
5. **Fuentes Citadas**: Cada respuesta devuelta incluye la procedencia exacta (archivo, página y fragmento de texto) de la información utilizada para responder.
