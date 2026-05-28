# Sistema RAG (Retrieval-Augmented Generation) Modular

Este proyecto proporciona un entorno y código base profesional para realizar consultas inteligentes sobre tus documentos PDF utilizando una arquitectura de generación aumentada por recuperación (RAG). 

## Tecnologías Utilizadas
- **LLM**: API de Groq con el modelo de inferencia ultrarrápido `llama-3.1-8b-instant` (vía `langchain-groq`).
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`) que se ejecuta localmente y de forma gratuita en tu CPU.
- **Base de Datos Vectorial**: ChromaDB local para almacenamiento de fragmentos.
- **Procesamiento de Archivos**: `pypdf` para lectura y procesamiento de documentos.

---

## Estructura del Proyecto

```text
Labrag/
├── documentacion.pdf        # Tu documento PDF de prueba
├── .env.example             # Plantilla de configuración de variables de entorno
├── requirements.txt         # Listado de dependencias necesarias
├── main.py                  # Código modular principal de la aplicación RAG
└── README.md                # Esta guía de uso
```

---

## Guía de Configuración Rápida (PowerShell en Windows)

Sigue estos pasos en tu terminal para poner en marcha el sistema:

### 1. Crear y Activar el Entorno Virtual
Abre tu terminal en el directorio del proyecto (`c:\Users\donat\Desktop\Labrag`) y ejecuta:

```powershell
# 1. Crear el entorno virtual en la carpeta .venv
python -m venv .venv

# 2. Activar el entorno virtual en Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

*(Si usas el símbolo del sistema clásico (CMD) de Windows, actívalo con: `.venv\Scripts\activate.bat`)*

### 2. Instalar las Dependencias
Con el entorno virtual activado, instala todas las librerías necesarias ejecutando:

```powershell
pip install -r requirements.txt
```

### 3. Configurar tu Clave de API de Groq
1. Crea una copia del archivo `.env.example` y llámalo `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Abre el archivo `.env` recién creado y reemplaza `tu_clave_api_de_groq_aqui` por tu clave de API real de Groq.
   ```env
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
   ```
   *Puedes obtener una clave de API de Groq totalmente gratis registrándote en [Groq Console](https://console.groq.com/).*

---

## Cómo Ejecutar el Proyecto

Una vez que hayas configurado tu `.env` con la API Key y con tu archivo `documentacion.pdf` en la carpeta raíz del proyecto, ejecuta el script principal:

```powershell
python main.py
```

### ¿Qué hace el script?
1. **Validación**: Verifica que la variable de entorno `GROQ_API_KEY` esté configurada.
2. **Carga Inteligente de Base Vectorial**: 
   - Si ya existe la base vectorial en la carpeta local `./chroma_db`, **la carga instantáneamente de disco** en menos de un segundo.
   - Si no existe (primera ejecución), procesa el PDF `Informe Final.pdf`, lo divide en fragmentos semánticos, calcula sus embeddings locales y los guarda en la base vectorial.
3. **Consola Interactiva de Preguntas (Bucle)**: El programa entra en un bucle interactivo de chat. Puedes escribir cualquier pregunta en tiempo real y el sistema te responderá basándose en el documento.
4. **Generación con Groq (llama-3.1-8b-instant)**: Recupera los fragmentos de contexto más relevantes del documento, los inyecta en el prompt optimizado y solicita al modelo en Groq redactar una respuesta precisa en español, indicando además las páginas fuente exactas y fragmentos citados.
5. **Salir**: Escribe `salir` o `exit` para terminar la sesión.
