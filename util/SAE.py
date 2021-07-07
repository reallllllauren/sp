from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers


def AE(data, output_dim, epoch, batch_size):
    x_train_simple = data.reshape(data.shape[0], -1)
    ENCODING_DIM_INPUT = x_train_simple.shape[1]
    ENCODING_DIM_OUTPUT = output_dim
    EPOCHS = epoch
    BATCH_SIZE = batch_size
    def sparse_AE(x_train):
        # input placeholder
        input_image = Input(shape=(ENCODING_DIM_INPUT,))
        encoded = Dense(500, activation='relu')(input_image)
        encoded = Dense(300, activation='relu')(encoded)
        encoded_output = Dense(ENCODING_DIM_OUTPUT)(encoded)
        decoded = Dense(200, activation='relu')(encoded_output)
        decoded = Dense(300, activation='relu')(decoded)
        decoded = Dense(500, activation='relu')(decoded)
        decoded_output = Dense(ENCODING_DIM_INPUT, activation='linear')(decoded)
        encoder = Model(input_image, encoded_output)
        autoencoder = Model(input_image, decoded_output)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

        return encoder, autoencoder

    encoder, autoencoder = sparse_AE(x_train=x_train_simple)
    data = encoder.predict(x_train_simple)
    return data

def SAE(data):
    data = AE(data, output_dim=400, epoch=100, batch_size=1024)
    data = AE(data, output_dim=300, epoch=100, batch_size=1024)
    data = AE(data, output_dim=200, epoch=100, batch_size=1024)
    data = AE(data, output_dim=200, epoch=100, batch_size=1024)
    return data
