from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import nltk
import numpy as np
import json
import random
import whisper
from werkzeug.utils import secure_filename
import os
import pickle
import requests
from io import BytesIO
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

# Inicialização
app = Flask(__name__)
CORS(app)
nltk.download('punkt')

FFMPEG_PATH = r"C:\\ffmpeg\\bin"
os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Carregar modelo Whisper
whisper_model = whisper.load_model("base")

def transcrever_audio(caminho):
    try:
        return whisper_model.transcribe(caminho)["text"]
    except:
        return "Erro na transcrição"

# Carregar intents e respostas
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}

# Carregar vocab e label encoder
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("labels.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)

# Carregar modelo treinado
model = load_model("model.h5")

lemmatizer = WordNetLemmatizer()

def bag_of_words(sentence):
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return np.array([1 if w in tokens else 0 for w in vocab])

# Busca locais
def encontrar_locais(lat, lon, tipo_local):
    if tipo_local not in ["hospital", "pharmacy"]:
        return "Tipo de local inválido. Use 'hospital' ou 'pharmacy'."

    q = f"""
    [out:json];
    (
        node["amenity"="{tipo_local}"](around:3000,{lat},{lon});
        way["amenity"="{tipo_local}"](around:3000,{lat},{lon});
        relation["amenity"="{tipo_local}"](around:3000,{lat},{lon});
    );
    out center tags;
    """
    print("Query enviada ao Overpass:")
    print(q)

    r = requests.post("http://overpass-api.de/api/interpreter", data={'data': q})
    if r.status_code != 200:
        print("Erro Overpass:")
        print(f"Status: {r.status_code}")
        print(f"Resposta: {r.text}")
        return f"Erro ao buscar {tipo_local}s."

    data = r.json()
    resultados = []

    for el in data["elements"]:
        tags = el.get("tags", {})
        nome = tags.get("name", f"{tipo_local.capitalize()} sem nome")
        rua = tags.get("addr:street", "")
        numero = tags.get("addr:housenumber", "")
        cidade = tags.get("addr:city", "")
        endereco = f"{rua}, {numero}, {cidade}".strip(', ').strip()

        # Coordenadas (latitude e longitude)
        if "center" in el:
            lat_local = el["center"].get("lat")
            lon_local = el["center"].get("lon")
        elif "lat" in el and "lon" in el:
            lat_local = el["lat"]
            lon_local = el["lon"]
        else:
            lat_local = lon_local = None

        if lat_local and lon_local:
            link_maps = f"https://www.google.com/maps?q={lat_local},{lon_local}"
            if endereco:
                resultado = f"{nome} - {endereco} (Lat: {lat_local}, Lon: {lon_local}) - {link_maps}"
            else:
                resultado = f"{nome} (Lat: {lat_local}, Lon: {lon_local}) - {link_maps}"
        else:
            resultado = f"{nome} - Coordenadas não disponíveis"

        resultados.append(resultado)

    tipo_cap = tipo_local.capitalize() + "s"
    return f"{tipo_cap} encontrados:\n" + "\n".join(resultados[:5]) if resultados else f"Nenhum {tipo_local} encontrado."

# Geração de áudio
def gerar_audio(texto):
    api_key = ""
    voice_id = ""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"

    headers = {
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    data = {
        "text": texto,
        "model_id": "eleven_multilingual_v2"
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return BytesIO(response.content)
        else:
            print("Erro na geração de áudio:", response.text)
            return None
    except Exception as e:
        print("Exceção ao gerar áudio:", e)
        return None

# Resposta
def responder(msg, lat=None, lon=None):
    print(f"Latitude: {lat}, Longitude: {lon}")
    bow = bag_of_words(msg)
    result = model.predict(np.array([bow]))[0]
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

    if tag in ["informar_localizacao", "buscar_hospital", "buscar_farmacia"] and lat and lon:
        tipo_local = "hospital" if tag != "buscar_farmacia" else "pharmacy"
        resposta = encontrar_locais(lat, lon, tipo_local)
    else:
        resposta = random.choice(responses.get(tag, ["Não entendi."]))

    return resposta, tag

# Rota texto
@app.route("/mensagem", methods=["POST"])
def mensagem():
    data = request.get_json()
    texto = data.get("mensagem")
    lat = data.get("lat")
    lon = data.get("lon")
    resposta, _ = responder(texto, lat, lon)
    return jsonify({"resposta": resposta})

# Rota áudio
@app.route("/mensagem_audio", methods=["POST"])
def mensagem_audio():
    if 'audio' not in request.files:
        return jsonify({"erro": "Sem áudio"}), 400

    arquivo = request.files['audio']
    lat = request.form.get("lat", type=float)
    lon = request.form.get("lon", type=float)

    if not arquivo.filename:
        return jsonify({"erro": "Arquivo inválido"}), 400

    path = os.path.join("uploads", secure_filename(arquivo.filename))
    os.makedirs("uploads", exist_ok=True)
    arquivo.save(path)

    texto = transcrever_audio(path)
    print("Texto reconhecido:", texto)

    resposta, tag = responder(texto, lat, lon)

    if tag == "informar_localizacao":
        return jsonify({"resposta": resposta})

    audio_bytes = gerar_audio(resposta)
    if audio_bytes:
        return send_file(audio_bytes, mimetype="audio/mpeg")
    else:
        return jsonify({"erro": "Falha ao gerar resposta em áudio"}), 500

# Início do servidor
if __name__ == "__main__":
    port = int(os.environ.get("PORT",
    5000))
    app.run(host='0.0.0.0', port=port)
