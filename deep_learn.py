#Criando uma rede neural para aprender padrao de escrita em números!

# %%
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


# %%
#Criando as bases de treino e teste
print('[INFO] acessing MNIST...')
((trainX, trainY), (testX, testY)) = mnist.load_data()


# %%
#Normalizando os dados da imagem, reenquadrando para o tamanho 28x28.
#  Faremos isso dividindo o conjunto por 255 (valor máximo de um pixel)

trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# %%
#Agora binarizaremos última camada

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# %%
#Definindo arquitetura da rede neural

model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# %%
#Calculando a função de custo com entropia cruzada categórica

sgd = SGD(0.01)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              epochs=100,
              batch_size=128)


# %%
#Comparando valores preditos com valores reais para checar a rede neural.

predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]))

# %%
#Exibindo evolução da rede até chegar nas métricas acima

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Época')
plt.ylabel('Função Custo/Acurácia')
plt.legend()
# %%
