# ZolpiTech

## Objetivo
ZolpiTech é um chatbot que tem como propósito ser um assistente completo para indivíduos que necessitam de apoio médico e orientação eficiente em áreas desconhecidas em tempo real. Ele foi projetado para:

1. **Recomendar profissionais e áreas da saúde próximos** com base nos problemas apresentados pelo indivíduo.
2. **Fornecer informações sobre disponibilidade** de médicos, postos de saúde, hospitais e farmácias.
3. **Auxiliar na escolha do atendimento mais proximo e eficaz**, especialmente em áreas desconhecidas.
4. **Utilizar reconhecimento de voz**, proporcionando acessibilidade para indivíduos com dificuldades de interação textual.

## Funcionalidades
- Identificar e recomendar locais abertos, como hospitais e farmácias, em tempo real.
- Verificar a disponibilidade de médicos e o tempo estimado de espera para atendimento.
- Direcionar o usuário ao local mais próximo e com menor fila.

## Recursos Utilizados
Para atender aos objetivos e funcionalidades descritos, as seguintes APIs podem ser integradas ao sistema:

- Tensorflow.
- Keras
- Whisper (OpenAI)
- nltk
- scikit-learn
- Python 
- Flask
- Flutter

## Progesso

Front-end: Utilizando Flutter construimos o Front-end que ja está quase pronto mas falta detalhes minímos como melhoria do chatbot e ligamento com API de localização para devolver os locais e mostrar em um mapa interativo.

Back-end: Para o back-end estamos utilizando python.

A arquitetura tambem usa respostas configuradas a partir de um arquivo intents.json, uma função de corespondência por similaridade textual e com classificação de intenções baseada em rede neural(Keras).



