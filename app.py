from fastapi import FastAPI, UploadFile, File
from starlette.responses import JSONResponse
import pandas as pd
import joblib
import io
import os

app = FastAPI()

# Carrega o modelo treinado
modelo = joblib.load("xgboost_model.pkl")

@app.get("/")
def raiz():
    return {"message": "API em execução"}

@app.post("/prever-csv/")
async def prever_csv(file: UploadFile = File(...)):
    # Lê o conteúdo do arquivo enviado
    conteudo = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(conteudo))

    # Colunas esperadas
    colunas_necessarias = ['price', 'earnings_ttm', 'marketcap', 'pe_ratio_ttm',
                           'revenue_ttm', 'total_shares', 'dividend_yield']

    # Validação
    if not all(col in df.columns for col in colunas_necessarias):
        return {"erro": "Colunas do CSV inválidas. Esperadas: " + ", ".join(colunas_necessarias)}

    # Previsão
    df['potencial_previsto'] = modelo.predict(df[colunas_necessarias])

    # Adiciona a coluna 'potencial_crescimento'
    def atribuir_categoria(potencial):
        if potencial == 0:
            return "Baixo"
        elif potencial == 1:
            return "Mediano"
        elif potencial == 2:
            return "Alto"
        else:
            return "Muito alto"

    df['potencial_crescimento'] = df['potencial_previsto'].apply(atribuir_categoria)

    # Converte o DataFrame para dicionário e retorna como JSON
    resultado_json = df.to_dict(orient='records')

    return JSONResponse(content=resultado_json)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
