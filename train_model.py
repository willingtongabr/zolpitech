import json
import nltk
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Iniciar o lemmatizer
lemmatizer = WordNetLemmatizer()

# Carregar intents.json
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

# Processar padrões
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((pattern, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatizar e remover duplicatas
words = [lemmatizer.lemmatize(w.lower()) for w in words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Criar treinamento
training_sentences = []
training_labels = []

for doc in documents:
    pattern_words = nltk.word_tokenize(doc[0])
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]

    bag = [0] * len(words)
    for s in pattern_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1

    training_sentences.append(bag)
    training_labels.append(doc[1])  # ← Aqui a label ainda está em string

# Codificar labels
lbl_encoder = LabelEncoder()
training_labels = lbl_encoder.fit_transform(training_labels)

# Arrays numpy
training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)

# Modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(training_sentences[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compilar
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Treinar
model.fit(training_sentences, training_labels, epochs=200, batch_size=5, verbose=1)

# Salvar modelo
model.save('model.h5')

# Salvar vocab e labels
with open("vocab.pkl", "wb") as f:
    pickle.dump(words, f)

with open("labels.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)

print("Modelo treinado e salvo com sucesso!")
