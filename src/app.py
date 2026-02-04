from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from dotenv import load_dotenv
import os
import pypdf
from typing import Optional, List, Dict, Any
from uuid import uuid4
from datetime import datetime

# =========================
# CONFIGURACIÓN INICIAL
# =========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="LEX Juris · CENDOJ Automation Prototype")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MAX_CHARS_SENTENCIA = 6000
MAX_FILES_LOTE = 10  # límite de seguridad para demo/TFG

# Memoria de sesiones en RAM (suficiente para TFG; no persistente)
SESIONES: Dict[str, Dict[str, Any]] = {}


# =========================
# PROMPT JURÍDICO BASE
# =========================

PROMPT_JURIDICO_BASE = (
    "Eres un asistente jurídico especializado en el análisis de resoluciones judiciales españolas. "
    "Tu función es extraer información jurídica relevante de forma objetiva y técnica. "
    "No emites dictámenes jurídicos ni valoraciones personales. "
    "No realizas inferencias no fundamentadas en el texto. "
    "Si una información no consta expresamente en la sentencia, debes indicarlo de forma clara. "
    "Debes limitar tu respuesta estrictamente a la tarea solicitada en cada consulta."
)

# =========================
# UTILIDADES
# =========================

def leer_pdf(file: UploadFile) -> str:
    """Extrae texto del PDF."""
    try:
        reader = pypdf.PdfReader(file.file)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text() or ""
        return texto
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el PDF: {str(e)}")


def construir_prompt(tipo: str, texto: str, pregunta: Optional[str] = None) -> str:
    """Prompts con restricciones (para que no devuelva 'todo')."""

    if tipo == "sujetos":
        return f"""
Identifica exclusivamente los sujetos intervinientes en la resolución judicial
y su posición procesal (demandante, demandado, acusado, Ministerio Fiscal,
Administración, etc.).

No incluyas hechos, fundamentos jurídicos, ratio decidendi ni fallo.

SENTENCIA:
{texto}
"""

    if tipo == "ratio":
        return f"""
Identifica exclusivamente la ratio decidendi de la sentencia.

Limita tu respuesta únicamente a la razón jurídica fundamental
que justifica el fallo.

No incluyas información sobre las partes, hechos ni el fallo.

SENTENCIA:
{texto}
"""

    if tipo == "normativa":
        return f"""
Identifica exclusivamente la normativa aplicada en la resolución judicial.

Incluye leyes, reglamentos o preceptos citados expresamente.
No incluyas hechos, análisis del caso, ratio decidendi ni fallo.

SENTENCIA:
{texto}
"""

    if tipo == "fallo":
        return f"""
Resume exclusivamente el fallo de la sentencia en una única frase clara y precisa.

No incluyas fundamentos jurídicos, hechos ni ratio decidendi.

SENTENCIA:
{texto}
"""

    if tipo == "consecuencia":
        return f"""
Identifica exclusivamente la consecuencia jurídica impuesta en la resolución
(pena, sanción, condena u obligación).

No incluyas otros aspectos de la sentencia.

SENTENCIA:
{texto}
"""

    if tipo == "otro":
        if not pregunta or not pregunta.strip():
            return """
Has seleccionado “Otra cuestión jurídica o fáctica”, pero no se ha proporcionado una pregunta.
Indica: “No se ha proporcionado pregunta”.
"""
        return f"""
Responde exclusivamente a la siguiente cuestión jurídica o fáctica,
basándote únicamente en el texto de la sentencia proporcionada.

Si la información no consta expresamente, indícalo de forma clara.

CUESTIÓN PLANTEADA:
{pregunta}

SENTENCIA:
{texto}
"""

    # Fallback de seguridad
    return f"""
Analiza jurídicamente la siguiente sentencia de forma objetiva.

SENTENCIA:
{texto}
"""


def llamar_modelo(prompt: str) -> str:
    """Centraliza la llamada al modelo para reutilizar en HTML/JSON/lote/sesiones."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PROMPT_JURIDICO_BASE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al llamar al modelo: {str(e)}")


def crear_sesion(texto_sentencia: str, nombre_archivo: str) -> str:
    """Crea una sesión para 'memoria' de una sentencia (en RAM)."""
    session_id = str(uuid4())
    SESIONES[session_id] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "archivo": nombre_archivo,
        "texto_sentencia": texto_sentencia,
        "consultas": []
    }
    return session_id


def registrar_consulta(session_id: str, tipo_analisis: str, pregunta: Optional[str], resultado: str):
    """Guarda histórico de consultas por sesión."""
    if session_id in SESIONES:
        SESIONES[session_id]["consultas"].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tipo_analisis": tipo_analisis,
            "pregunta": pregunta,
            "resultado": resultado
        })


# =========================
# VISTA PRINCIPAL (GET)
# =========================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "resultado": None}
    )


# =========================
# PROCESAMIENTO HTML (POST)
# =========================

@app.post("/analizar", response_class=HTMLResponse)
async def analizar(
    request: Request,
    tipo_analisis: str = Form(...),
    archivo: UploadFile = File(...),
    pregunta: Optional[str] = Form(None),
):
    texto = leer_pdf(archivo)
    texto_recortado = texto[:MAX_CHARS_SENTENCIA]
    prompt = construir_prompt(tipo_analisis, texto_recortado, pregunta)
    resultado = llamar_modelo(prompt)

    # (Plus) crea sesión automáticamente para reutilización futura (opcional)
    session_id = crear_sesion(texto_recortado, archivo.filename)
    registrar_consulta(session_id, tipo_analisis, pregunta, resultado)

    # Puedes mostrar session_id en pantalla si quieres (no obligatorio)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "resultado": resultado, "session_id": session_id}
    )


# =========================
# PROCESAMIENTO JSON (POST)
# =========================

@app.post("/analizar_json", response_class=JSONResponse)
async def analizar_json(
    tipo_analisis: str = Form(...),
    archivo: UploadFile = File(...),
    pregunta: Optional[str] = Form(None),
):
    texto = leer_pdf(archivo)
    texto_recortado = texto[:MAX_CHARS_SENTENCIA]
    prompt = construir_prompt(tipo_analisis, texto_recortado, pregunta)
    resultado = llamar_modelo(prompt)

    session_id = crear_sesion(texto_recortado, archivo.filename)
    registrar_consulta(session_id, tipo_analisis, pregunta, resultado)

    return {
        "session_id": session_id,
        "tipo_analisis": tipo_analisis,
        "archivo": archivo.filename,
        "resultado": resultado
    }


# =========================
# (PLUS) ANÁLISIS EN LOTE (JSON)
# =========================
# Permite subir varios PDFs y aplicar el mismo tipo de análisis a todos.
# Ideal para demostrar "automatización" + comparación jurisprudencial.

@app.post("/analizar_lote", response_class=JSONResponse)
async def analizar_lote(
    tipo_analisis: str = Form(...),
    archivos: List[UploadFile] = File(...),
    pregunta: Optional[str] = Form(None),
):
    if len(archivos) > MAX_FILES_LOTE:
        raise HTTPException(status_code=400, detail=f"Máximo {MAX_FILES_LOTE} archivos por lote.")

    resultados = []
    for a in archivos:
        texto = leer_pdf(a)[:MAX_CHARS_SENTENCIA]
        prompt = construir_prompt(tipo_analisis, texto, pregunta)
        res = llamar_modelo(prompt)
        resultados.append({
            "archivo": a.filename,
            "resultado": res
        })

    return {
        "tipo_analisis": tipo_analisis,
        "numero_sentencias": len(archivos),
        "resultados": resultados
    }


# =========================
# (PLUS) CONTINUAR ANÁLISIS SOBRE UNA SENTENCIA (SESIÓN)
# =========================
# Permite "memoria" controlada por session_id sin re-subir el PDF.

@app.post("/analizar_sesion", response_class=JSONResponse)
async def analizar_sesion(
    session_id: str = Form(...),
    tipo_analisis: str = Form(...),
    pregunta: Optional[str] = Form(None),
):
    if session_id not in SESIONES:
        raise HTTPException(status_code=404, detail="Sesión no encontrada o caducada.")

    texto = SESIONES[session_id]["texto_sentencia"]
    prompt = construir_prompt(tipo_analisis, texto, pregunta)
    resultado = llamar_modelo(prompt)

    registrar_consulta(session_id, tipo_analisis, pregunta, resultado)

    return {
        "session_id": session_id,
        "tipo_analisis": tipo_analisis,
        "archivo": SESIONES[session_id]["archivo"],
        "resultado": resultado,
        "consultas_previas": len(SESIONES[session_id]["consultas"])
    }


# =========================
# (PLUS) VER HISTÓRICO DE UNA SESIÓN (JSON)
# =========================

@app.get("/sesion/{session_id}", response_class=JSONResponse)
async def ver_sesion(session_id: str):
    if session_id not in SESIONES:
        raise HTTPException(status_code=404, detail="Sesión no encontrada o caducada.")
    return SESIONES[session_id]
