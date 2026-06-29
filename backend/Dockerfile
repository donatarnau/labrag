# Stage 1: Builder para instalar dependencias y pre-descargar el modelo
FROM python:3.11-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema necesarias para construir algunas librerías
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir la ruta global para el cache de Hugging Face
ENV HF_HOME=/cache

# Copiar archivos de dependencias
COPY requirements.txt .

# 1. Optimización: Instalar la versión de PyTorch orientada a CPU en lugar de la versión por defecto con CUDA.
# Esto reduce el tamaño de la imagen de ~3GB a ~600MB y evita problemas de límite de memoria (OOM) en Docker.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Instalar el resto de dependencias de Python
RUN pip install --no-cache-dir --user -r requirements.txt

# Pre-descargar el modelo de HuggingFace en el directorio global /cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Stage 2: Runtime limpio
FROM python:3.11-slim AS runner

WORKDIR /app

# Crear un usuario de sistema no privilegiado para ejecutar la aplicación
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -s /sbin/nologin appuser

# Definir la ruta global para el cache de Hugging Face en el runner
ENV HF_HOME=/cache
# 2. Robustez: Forzar a Hugging Face a operar 100% offline.
# Evita que intente conectarse al Hub para verificar actualizaciones, lo cual causa bloqueos o fallos en Docker sin internet.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copiar las dependencias instaladas en el builder con permisos adecuados
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copiar el modelo de HuggingFace pre-descargado
COPY --from=builder --chown=appuser:appuser /cache /cache
# 3. Permisos: Permitir que cualquier usuario (incluso si se ejecuta con --user en Docker) pueda leer/escribir en el cache.
RUN chmod -R 777 /cache

# Copiar el código de la aplicación y el documento (PDF o TXT)
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser documentacion.* .

# Crear el directorio para ChromaDB y asignar permisos
RUN mkdir -p /app/chroma_db && chown -R appuser:appuser /app

# Cambiar al usuario no privilegiado por defecto
USER appuser

EXPOSE 8000

# Comando por defecto para ejecutar la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
