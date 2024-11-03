from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
# Habilitar CORS apenas para o domínio da sua aplicação Next.js
CORS(app, resources={r"/prever": {"origins": "https://nextjs-sprint-d63r.vercel.app"}})

# Carregar o modelo e o vetorizador TF-IDF
with open('modelo_random_forest2.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as v:
    vetorizador = pickle.load(v)

@app.route('/prever', methods=['POST'])
def prever():
    # Recebe o parâmetro do sintoma via JSON no corpo da requisição
    dados = request.get_json()

    # Verifica se o parâmetro "Sintoma" foi enviado
    if not dados or 'Sintoma' not in dados:
        return jsonify({'erro': 'O campo "Sintoma" é obrigatório.'}), 400

    parametro = dados['Sintoma']

    try:
        # Coloca o parametro em uma lista e vetoriza usando o vetorizador TF-IDF
        entrada_vetorizada = vetorizador.transform([parametro])

        # Realiza a previsão com o modelo carregado
        resultado = modelo.predict(entrada_vetorizada)

        # Retorna a previsão como JSON
        return jsonify({'previsao': resultado[0]})
    except Exception as e:
        # Captura qualquer erro e retorna uma mensagem amigável
        return jsonify({'erro': f'Erro ao processar a previsão: {str(e)}'}), 500

if __name__ == "__main__":
    # Torna o servidor acessível externamente
    app.run(host="0.0.0.0")
