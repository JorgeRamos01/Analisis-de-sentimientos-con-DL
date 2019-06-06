#Importamos todos los modulos importantes para generar nuestro modelo
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing import sequence
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os

#Hacemos un pequeno analisis exploratorio de nuestros datos
#para saber cuantas palabras unicas hay en nuestro corpus y
#cuantas palabras hay en cada resena

maxlen=0
word_freqs=collections.Counter()
num_recs=0
ftrain=open(os.path.join("/home/jorge/Documentos/Datos_complejos/Proyecto_final/DATA_DIR", "twits.txt"), 'r') #twits_limpio.txt
for line in ftrain:
	sentence,label = line.strip().split("\t")
	words=nltk.word_tokenize(sentence.lower())
	if len(words)>maxlen:
		maxlen=len(words)
	for word in words:
		word_freqs[word] +=1
	num_recs +=1
ftrain.close()

print(maxlen)
print(len(word_freqs))

MAX_FEATURES=17000
MAX_SENTENCE_LENGTH =50

#La entrada para la RNN son palabras indexadas con su numero de frecuencia en el documento, ademas consideramos 2 etiquetas para palabras que no se encuentran en el corpus
vocab_size = min(MAX_FEATURES, len(word_freqs))+2
word2index = {x[0]: i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"]=0
word2index["UNK"]=1
index2word={v:k for k, v in word2index.items()}

#Preparamos nuestros datos para darlos como entrada en la RNN
X=np.empty((num_recs,),dtype=list)
y=np.zeros((num_recs,))
i=0
ftrain=open(os.path.join("/home/jorge/Documentos/Datos_complejos/Proyecto_final/DATA_DIR", "twits.txt"), 'r')
for line in ftrain:
	sentence, label = line.strip().split("\t")
	words=nltk.word_tokenize(sentence.lower())
	seqs=[]
	for word in words:
		if word in word2index:
			seqs.append(word2index[word])
		else:
			seqs.append(word2index["UNK"])
	X[i]=seqs
	y[i]=int(label)
	i += 1
ftrain.close()
X=sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

#Cortamos los datos etiquetados en un conjunto de entrenamiento y
#en uno de prueba en proporcion 80-20

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

#Generamos la RNN a usar y establecemos los parametros (ajustables)
EMBEDDIND_SIZE=128
HIDDEN_LAYER_SIZE=256
BATCH_SIZE=60
NUM_EPOCHS =10

model=Sequential()
model.add(Embedding(vocab_size, EMBEDDIND_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))   #Establecemos una capa de dropout de 20%
model.add(Dense(1))
model.add(Activation("sigmoid"))  #Dado que son datos binarios (Positivo-Negativo) usamos la funcion sigmoide para clasificar
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#rms= optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=["accuracy"])
#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])   #Compilamos el modelo con un optimizador tipo ADAM y funcion de perdida binary-cross entropy (respuesta binaria)
print(model.summary())                                       #Mostramos las especificaciones del modelo a generar

history=model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))   #Aplicamos el modelo usando los datos de entrenamiento y validamos usando los datos de prueba que generamos

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)    #Guardamos nuestros valores de precision y error obtenidos
Y_pred = model.predict(Xtest)
print('Confusion Matrix')
print(confusion_matrix((Y_pred>0.5).astype(int), ytest))
print('Classification Report')
target_names = ['Positivo', 'Negativo']
print(classification_report(ytest, (Y_pred>0.5).astype(int), target_names=target_names))
print("\n\n")
print("Test score: %.3f, Precisión: %.3f" % (score, acc))