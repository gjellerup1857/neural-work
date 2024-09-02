from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorial


X_train, X_test, y_train, y_test = get_train_text()
X_train = X_train.reshape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)


y_train_hot = to_categorial(y_train)
y_test_hot = to_categorial(y_test)
print("X_train.shape=", X_train.shape)


model = Sequential()
model.add.(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 11, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
model.complie(loss=keras.losses.categorial_crossentropy,optimize=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(X_train, y_train_hot, batch_size=100, epochs=200, verbose=1, validation_data=(X_test, y_test_hot))


X_train = X_train.reshaape(X_train.shape[0], 20, 11, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 11, 1)
sorce = model.evaluate(X_test, y_test_hot, verbose=1)


from keras.models import load_model
model.save('ASR.h5')


mfcc = wav2mfcc('./data/happy/012c8314_nohash_0.wav')
mfcc_reshaped = mfcc.reshape(1, 20, 11, 1)
print("labels=", get_labels())
print("predict=", np.argmax(model.predict(mfcc_reshaped)))