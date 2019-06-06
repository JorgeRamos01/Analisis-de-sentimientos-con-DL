#Importamos todos los modulos importantes para generar nuestro modelo
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
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
ftrain=open(os.path.join("/home/jorge/Documentos/Datos_complejos/Proyecto_final/DATA_DIR", "amazon_cells_labelled.txt"), 'r')
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


MAX_FEATURES=2000
MAX_SENTENCE_LENGTH =30

vocab_size = min(MAX_FEATURES, len(word_freqs))+2
word2index = {x[0]: i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"]=0
word2index["UNK"]=1
index2word={v:k for k, v in word2index.items()}

X=np.empty((num_recs,),dtype=list)
y=np.zeros((num_recs,))
i=0
ftrain=open(os.path.join("/home/jorge/Documentos/Datos_complejos/Proyecto_final/DATA_DIR", "amazon_cells_labelled.txt"), 'r')
for line in ftrain:
	sentence,label = line.strip().split("\t")
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

EMBEDDIND_SIZE=512
HIDDEN_LAYER_SIZE=128
BATCH_SIZE=32
NUM_EPOCHS =15

model=Sequential()
model.add(Embedding(vocab_size, EMBEDDIND_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

history=model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
Y_pred = model.predict(Xtest)
print('Confusion Matrix')
print(confusion_matrix((Y_pred>0.5).astype(int), ytest))
print('Classification Report')
target_names = ['Positivo', 'Negativo']
print(classification_report(ytest, (Y_pred>0.5).astype(int), target_names=target_names))
print("\n\n")
print("Test score: %.3f, Precisión: %.3f" % (score, acc))

plt.subplot(211)
plt.title("Nivel de Precisión")
plt.plot(history.history["acc"], color="g", label="Entrenamiento")
plt.plot(history.history["val_acc"], color="b", label="Validación")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Pérdida")
plt.plot(history.history["loss"], color="g", label="Entrenamiento")
plt.plot(history.history["val_loss"], color="b", label="Validación")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

