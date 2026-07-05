# 🤖 Asistente Virtual de RRHH - NovaTech Solutions

Este repositorio contiene el código fuente y la arquitectura de despliegue del **Proyecto Integrador** para el **Diploma AI Engineer** de **DMC Institute**.

Se ha desarrollado un **Asistente Virtual con Arquitectura RAG (Retrieval-Augmented Generation)** optimizado para responder de forma automática, inmediata y precisa las consultas frecuentes de los colaboradores de **NovaTech Solutions** relacionadas con el **Manual del Empleado** y las políticas internas de la empresa (vacaciones, licencias, capacitaciones, beneficios, seguridad de la información, entre otros).

> 🚀 **Aplicación en producción**
>
> 🔗 **[Probar la aplicación en Streamlit Cloud](https://proyectochatbotrrhh-r3h5fvj7aw5jv2wpz8qlkf.streamlit.app)**
>

---

# 🏗️ Arquitectura Técnica

El sistema implementa una arquitectura **RAG (Retrieval-Augmented Generation)** que permite minimizar las alucinaciones de los modelos de lenguaje al responder únicamente con información recuperada desde el Manual del Empleado de NovaTech Solutions.

## Componentes de la solución

| Componente | Tecnología |
|------------|------------|
| Interfaz Web | Streamlit |
| Framework LLM | LangChain |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Base de datos vectorial | ChromaDB |
| Modelo LLM | Llama 3.1 8B (`llama-3.1-8b-instant`) |
| API de Inferencia | Groq |
| Gestión de variables | python-dotenv |

---

## 📌 Flujo de la Arquitectura

```text
                 ┌──────────────────────────┐
                 │ Manual del Empleado PDF  │
                 └────────────┬─────────────┘
                              │
                              ▼
                    PyPDFLoader (LangChain)
                              │
                              ▼
     RecursiveCharacterTextSplitter (Chunking)
                              │
                              ▼
      HuggingFaceEmbeddings (all-MiniLM-L6-v2)
                              │
                              ▼
                     Base Vectorial ChromaDB
                              │
                 Búsqueda por Similaridad
                              │
                              ▼
             Contexto Recuperado (Top-K Chunks)
                              │
                              ▼
             Prompt + Llama 3.1 8B (Groq API)
                              │
                              ▼
              Interfaz Web Streamlit (Chat)
```

---

# ⚙️ Tecnologías Utilizadas

- 🐍 Python 3.11+
- 🎈 Streamlit
- 🦜 LangChain
- 🤗 Hugging Face Embeddings
- 📚 ChromaDB
- ⚡ Groq API
- 🦙 Llama 3.1 8B Instant
- 📄 PyPDFLoader
- 🔐 python-dotenv

---

# 🧠 Funcionamiento del Sistema RAG

El flujo completo del asistente consta de las siguientes etapas:

### 1. Carga del documento

El Manual del Empleado en formato PDF es procesado utilizando:

```python
PyPDFLoader
```

---

### 2. Segmentación (Chunking)

El documento se divide utilizando:

```python
RecursiveCharacterTextSplitter
```

Configuración utilizada:

| Parámetro | Valor |
|-----------|-------|
| chunk_size | 1000 |
| chunk_overlap | 200 |

Esta configuración permite conservar el contexto de tablas y párrafos relacionados.

---

### 3. Generación de Embeddings

Cada fragmento del documento es convertido en un vector semántico mediante:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Ventajas:

- Gratuito
- Open Source
- Muy rápido
- Excelente calidad semántica

---

### 4. Base de Datos Vectorial

Los embeddings son almacenados en una base vectorial local utilizando:

```text
ChromaDB
```

Esto permite realizar búsquedas por similitud semántica con baja latencia.

---

### 5. Recuperación de Contexto

Cuando el usuario realiza una consulta:

1. La pregunta se convierte en embedding.
2. Se buscan los fragmentos más similares.
3. Los resultados recuperados son enviados al modelo LLM.

---

### 6. Generación de la Respuesta

El modelo:

```text
Llama 3.1 8B
```

es consumido mediante:

```text
Groq API
```

La respuesta se construye únicamente utilizando el contexto recuperado desde el manual.

---

# 🛡️ Control de Alucinaciones

Uno de los principales objetivos del proyecto es evitar respuestas inventadas.

Para ello se implementó un **System Prompt** restrictivo que obliga al modelo a responder únicamente con información presente en el documento.

Si no existe contexto suficiente, el chatbot responde con el siguiente mensaje:

```text
Lo siento, no tengo esa información.
Por favor contacta a RRHH directamente.
```

Esta estrategia reduce significativamente las respuestas incorrectas propias de los modelos generativos.

---

# ⚙️ Instalación Local

## 1. Clonar el repositorio

```bash
git clone https://github.com/jonyJE/proyecto_chatbot_rrhh.git

cd proyecto_chatbot_rrhh
```

---

## 2. Crear un entorno virtual

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 4. Configurar variables de entorno

Crear un archivo llamado:

```text
.env
```

Agregar la siguiente variable:

```env
GROQ_API_KEY="gsk_tu_llave_secreta"
```

> **Importante**
>
> El archivo `.env` se encuentra incluido dentro del `.gitignore`, evitando publicar credenciales privadas en GitHub.

---

## 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

Una vez iniciada, Streamlit mostrará una dirección similar a:

```text
http://localhost:8501
```

---

# 📸 Casos de Uso

El asistente fue validado utilizando diferentes escenarios.

## ✅ Consulta directa

**Pregunta**

> ¿Cuál es el presupuesto anual para capacitaciones?

**Respuesta**

> S/. 500 por año.

---

## ✅ Consulta basada en reglas

**Pregunta**

> Llevo 6 años trabajando en la empresa.

> ¿Cuántos días de vacaciones me corresponden?

**Respuesta**

> 30 días.

---

## ✅ Consulta fuera del contexto

**Pregunta**

> ¿Cuál será el menú del comedor mañana?

**Respuesta**

```text
Lo siento, no tengo esa información.
Por favor contacta a RRHH directamente.
```

---

# 📂 Estructura del Proyecto

```text
proyecto_chatbot_rrhh/

│

├── app.py

├── requirements.txt

├── .env

├── .gitignore

├── README.md

└──manual_rrhh_novatech.pdf

```

---

# 🚀 Características

- Arquitectura RAG
- Recuperación semántica
- Embeddings Open Source
- LLM de baja latencia mediante Groq
- Control de alucinaciones
- Interfaz conversacional
- Historial de conversación
- Base vectorial local
- Fácil despliegue en Streamlit Cloud

---

# 📈 Mejoras Futuras

- Autenticación mediante Azure AD
- Historial persistente en base de datos
- Soporte para múltiples documentos
- Citas automáticas indicando la página del manual
- Integración con Microsoft Teams
- Panel administrativo para RRHH
- Registro de métricas y analítica de consultas
- Soporte para múltiples idiomas

---

# 👨‍💻 Créditos

**Desarrollador**

**Infanzon Castro Jonathan**
**Nepo Chavez Angel Arturo**

AI Engineer • Data Scientist

---

**Institución**

DMC Institute

---

**Programa Académico**

Diploma Avanzado AI Engineer (2026)

---

# 📄 Licencia

Este proyecto fue desarrollado con fines académicos como parte del **Proyecto Integrador del Diploma AI Engineer**.

El contenido del Manual del Empleado corresponde a una empresa ficticia (**NovaTech Solutions**) creada exclusivamente para fines educativos y de demostración tecnológica.

---
