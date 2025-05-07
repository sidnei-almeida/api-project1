from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
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

    # Mapeia os valores numéricos da coluna 'potencial_previsto' para as categorias
    def mapear_potencial_crescimento(valor):
        if valor == 0:
            return "Baixo"
        elif valor == 1:
            return "Mediano"
        elif valor == 2:
            return "Alto"
        elif valor == 3:
            return "Muito alto"
        else:
            return "Desconhecido"  # Caso algum valor não esperado seja previsto

    df['potencial_crescimento'] = df['potencial_previsto'].apply(mapear_potencial_crescimento)

    # Converte para CSV em memória
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Retorna como resposta para download
    return StreamingResponse(buffer, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=previsoes.csv"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
