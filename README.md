# 📈 API de Previsão de Potencial de Crescimento com FastAPI

Este projeto implementa uma API REST com **FastAPI** para servir um modelo de *Machine Learning* treinado para prever o **Potencial de Crescimento (PC)** de empresas situadas no continente americano. A previsão é feita com base em dados financeiros e macroeconômicos, utilizando técnicas de classificação supervisionada.

---

## 🚀 Funcionalidades

- 🔮 Previsão do Potencial de Crescimento via endpoint `/predict`
- 📊 Entrada baseada em indicadores como:
  - Dividend Yield
  - Market Cap
  - P/E Ratio
- 🧠 Modelo treinado com *XGBoost* e classes definidas por *KMeans*
- 🔒 Validação de dados com Pydantic
- 🧪 Testes de integridade da API
- 🌐 Pronto para deploy com Uvicorn

---

## 🧰 Tecnologias utilizadas

- Python 3.11+
- FastAPI
- Scikit-Learn
- XGBoost
- Pandas & NumPy
- Pydantic
- Uvicorn

---

## 📦 Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/sidnei-almeida/api-project1.git
cd api-project1
pip install -r requirements.txt
