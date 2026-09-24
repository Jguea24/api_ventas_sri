# ============================================================
# API DE PREDICCIÓN DE VENTAS - SRI
# MODELO ORIGINAL
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

MODELO_PATH = "modelo_ventas_api.joblib"

app = FastAPI(
    title="API de Predicción de Ventas SRI",
    description="API para predecir las ventas del siguiente período",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CARGAR MODELO
# ============================================================

modelo = None

try:

    modelo = joblib.load(MODELO_PATH)

    print("=" * 70)
    print("MODELO CARGADO CORRECTAMENTE")
    print("=" * 70)
    print(f"Archivo: {MODELO_PATH}")
    print(f"Tipo: {type(modelo)}")

    # Mostrar las variables que espera el modelo
    if hasattr(modelo, "feature_names_in_"):

        print("\nVARIABLES ESPERADAS POR EL MODELO:")
        for i, variable in enumerate(modelo.feature_names_in_, start=1):
            print(f"{i:02d}. {variable}")

    print("=" * 70)


except Exception as e:

    print("=" * 70)
    print("ERROR AL CARGAR EL MODELO")
    print("=" * 70)
    print(f"Detalle: {e}")
    print("=" * 70)


# ============================================================
# MODELO DE DATOS DE ENTRADA
# ============================================================

class DatosVenta(BaseModel):

    # Información temporal
    MES: int

    # Ubicación y sector
    CODIGO_SECTOR_N1: int
    PROVINCIA: str
    CANTON: str

    # Ventas
    VENTAS_NETAS_TARIFA_GRAVADA: float
    VENTAS_NETAS_TARIFA_0: float
    VENTAS_NETAS_TARIFA_VARIABLE: float
    VENTAS_NETAS_TARIFA_5: float

    # Comercio exterior
    EXPORTACIONES: float

    # Compras
    COMPRAS_NETAS_TARIFA_GRAVADA: float
    COMPRAS_NETAS_TARIFA_0: float
    IMPORTACIONES: float
    COMPRAS_RISE: float

    # Totales
    TOTAL_COMPRAS: float
    TOTAL_VENTAS: float

    # Historial
    VENTAS_MES_ANTERIOR: float


# ============================================================
# RUTA PRINCIPAL
# ============================================================

@app.get("/")
def inicio():

    if os.path.exists("index.html"):
        return FileResponse("index.html")

    return {
        "mensaje": "API de Predicción de Ventas SRI funcionando correctamente",
        "modelo": MODELO_PATH
    }


# ============================================================
# RUTA DE SALUD
# ============================================================

@app.get("/salud")
def salud():

    return {
        "estado": "activo",
        "modelo_cargado": modelo is not None,
        "modelo": MODELO_PATH
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post("/predecir")
def predecir(datos: DatosVenta):

    print("\n" + "=" * 70)
    print("DATOS RECIBIDOS PARA PREDICCIÓN")
    print("=" * 70)

    print(f"MES = {datos.MES}")
    print(f"CODIGO_SECTOR_N1 = {datos.CODIGO_SECTOR_N1}")
    print(f"PROVINCIA = '{datos.PROVINCIA}'")
    print(f"CANTON = '{datos.CANTON}'")

    print(f"VENTAS_NETAS_TARIFA_GRAVADA = {datos.VENTAS_NETAS_TARIFA_GRAVADA}")
    print(f"VENTAS_NETAS_TARIFA_0 = {datos.VENTAS_NETAS_TARIFA_0}")
    print(f"VENTAS_NETAS_TARIFA_VARIABLE = {datos.VENTAS_NETAS_TARIFA_VARIABLE}")
    print(f"VENTAS_NETAS_TARIFA_5 = {datos.VENTAS_NETAS_TARIFA_5}")

    print(f"EXPORTACIONES = {datos.EXPORTACIONES}")

    print(f"COMPRAS_NETAS_TARIFA_GRAVADA = {datos.COMPRAS_NETAS_TARIFA_GRAVADA}")
    print(f"COMPRAS_NETAS_TARIFA_0 = {datos.COMPRAS_NETAS_TARIFA_0}")
    print(f"IMPORTACIONES = {datos.IMPORTACIONES}")
    print(f"COMPRAS_RISE = {datos.COMPRAS_RISE}")

    print(f"TOTAL_COMPRAS = {datos.TOTAL_COMPRAS}")
    print(f"TOTAL_VENTAS = {datos.TOTAL_VENTAS}")

    print(f"VENTAS_MES_ANTERIOR = {datos.VENTAS_MES_ANTERIOR}")


    # ========================================================
    # VALIDACIONES
    # ========================================================

    if modelo is None:

        return {
            "success": False,
            "error": "El modelo no está cargado."
        }


    if datos.MES < 1 or datos.MES > 12:

        return {
            "success": False,
            "error": "El MES debe estar entre 1 y 12."
        }


    if datos.CODIGO_SECTOR_N1 < 1:

        return {
            "success": False,
            "error": "El código del sector debe ser válido."
        }


    # ========================================================
    # CALCULAR AÑO
    # ========================================================

    año_actual = 2026


    # ========================================================
    # CALCULAR MES SIGUIENTE
    # ========================================================

    mes_siguiente = datos.MES + 1
    año_siguiente = año_actual

    if mes_siguiente == 13:

        mes_siguiente = 1
        año_siguiente = 2027


    # ========================================================
    # CALCULAR VARIACIÓN DE VENTAS
    # ========================================================

    variacion_ventas = (
        datos.TOTAL_VENTAS
        - datos.VENTAS_MES_ANTERIOR
    )


    # ========================================================
    # CREAR DATAFRAME
    # ========================================================

    entrada = pd.DataFrame([{

        "AÑO": año_actual,

        "MES": datos.MES,

        "CODIGO_SECTOR_N1": datos.CODIGO_SECTOR_N1,

        "PROVINCIA": datos.PROVINCIA,

        "CANTON": datos.CANTON,

        "VENTAS_NETAS_TARIFA_GRAVADA":
            datos.VENTAS_NETAS_TARIFA_GRAVADA,

        "VENTAS_NETAS_TARIFA_0":
            datos.VENTAS_NETAS_TARIFA_0,

        "VENTAS_NETAS_TARIFA_VARIABLE":
            datos.VENTAS_NETAS_TARIFA_VARIABLE,

        "VENTAS_NETAS_TARIFA_5":
            datos.VENTAS_NETAS_TARIFA_5,

        "EXPORTACIONES":
            datos.EXPORTACIONES,

        "COMPRAS_NETAS_TARIFA_GRAVADA":
            datos.COMPRAS_NETAS_TARIFA_GRAVADA,

        "COMPRAS_NETAS_TARIFA_0":
            datos.COMPRAS_NETAS_TARIFA_0,

        "IMPORTACIONES":
            datos.IMPORTACIONES,

        "COMPRAS_RISE":
            datos.COMPRAS_RISE,

        "TOTAL_COMPRAS":
            datos.TOTAL_COMPRAS,

        "TOTAL_VENTAS":
            datos.TOTAL_VENTAS,

        "MES_SIGUIENTE":
            mes_siguiente,

        "AÑO_SIGUIENTE":
            año_siguiente,

        "VENTAS_MES_ANTERIOR":
            datos.VENTAS_MES_ANTERIOR,

        "VARIACION_VENTAS":
            variacion_ventas

    }])


    # ========================================================
    # CORRECCIÓN DE TIPOS
    # ========================================================

    entrada["PROVINCIA"] = (
        entrada["PROVINCIA"]
        .astype(str)
        .astype(object)
    )

    entrada["CANTON"] = (
        entrada["CANTON"]
        .astype(str)
        .astype(object)
    )


    # ========================================================
    # ASEGURAR VARIABLES NUMÉRICAS
    # ========================================================

    columnas_numericas = [

        "AÑO",
        "MES",
        "CODIGO_SECTOR_N1",

        "VENTAS_NETAS_TARIFA_GRAVADA",
        "VENTAS_NETAS_TARIFA_0",
        "VENTAS_NETAS_TARIFA_VARIABLE",
        "VENTAS_NETAS_TARIFA_5",

        "EXPORTACIONES",

        "COMPRAS_NETAS_TARIFA_GRAVADA",
        "COMPRAS_NETAS_TARIFA_0",

        "IMPORTACIONES",
        "COMPRAS_RISE",

        "TOTAL_COMPRAS",
        "TOTAL_VENTAS",

        "MES_SIGUIENTE",
        "AÑO_SIGUIENTE",

        "VENTAS_MES_ANTERIOR",
        "VARIACION_VENTAS"
    ]


    for columna in columnas_numericas:

        entrada[columna] = pd.to_numeric(
            entrada[columna],
            errors="raise"
        )


    # ========================================================
    # ORDEN EXACTO DEL MODELO ORIGINAL
    # ========================================================

    columnas_modelo = [

        "AÑO",
        "MES",
        "CODIGO_SECTOR_N1",
        "PROVINCIA",
        "CANTON",

        "VENTAS_NETAS_TARIFA_GRAVADA",
        "VENTAS_NETAS_TARIFA_0",
        "VENTAS_NETAS_TARIFA_VARIABLE",
        "VENTAS_NETAS_TARIFA_5",

        "EXPORTACIONES",

        "COMPRAS_NETAS_TARIFA_GRAVADA",
        "COMPRAS_NETAS_TARIFA_0",

        "IMPORTACIONES",
        "COMPRAS_RISE",

        "TOTAL_COMPRAS",
        "TOTAL_VENTAS",

        "MES_SIGUIENTE",
        "AÑO_SIGUIENTE",

        "VENTAS_MES_ANTERIOR",
        "VARIACION_VENTAS"
    ]


    entrada = entrada[columnas_modelo]


    # ========================================================
    # MOSTRAR DATAFRAME
    # ========================================================

    print("\n" + "=" * 70)
    print("DATAFRAME ENVIADO AL MODELO")
    print("=" * 70)

    print(entrada)

    print("\nDIMENSIONES:")
    print(entrada.shape)

    print("\nTIPOS DE DATOS:")
    print(entrada.dtypes)


    # ========================================================
    # REALIZAR PREDICCIÓN
    # ========================================================

    try:

        prediccion = modelo.predict(entrada)

        ventas_predichas = float(prediccion[0])


        print("\n" + "=" * 70)
        print("PREDICCIÓN REALIZADA CORRECTAMENTE")
        print("=" * 70)

        print(
            f"Ventas predichas para el siguiente período: "
            f"{ventas_predichas:,.2f}"
        )


        # ====================================================
        # RESPUESTA
        # ====================================================

        return {

            "success": True,

            "ventas_predichas": round(
                ventas_predichas,
                2
            ),

            "año_actual": año_actual,

            "mes_actual": datos.MES,

            "mes_siguiente": mes_siguiente,

            "año_siguiente": año_siguiente,

            "variacion_ventas": round(
                variacion_ventas,
                2
            ),

            "provincia": datos.PROVINCIA,

            "canton": datos.CANTON,

            "sector": datos.CODIGO_SECTOR_N1
        }


    except Exception as e:

        print("\n" + "=" * 70)
        print("ERROR EN LA PREDICCIÓN")
        print("=" * 70)

        print(f"Tipo: {type(e).__name__}")
        print(f"Detalle: {e}")


        return {

            "success": False,

            "error": type(e).__name__,

            "detalle": str(e)

        }


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )