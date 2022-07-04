from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.sequence import pad_sequences

class code2summary():

    def __init__(self, latent_dim, embedding_dim, max_code_len, max_summary_len, x_vocab, y_vocab, x_tokenizer, y_tokenizer):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.max_code_len = max_code_len
        self.max_summary_len = max_summary_len
        self.x_vocab = x_vocab
        self.y_vocab = y_vocab
        self.x_tokenizer = x_tokenizer
        self.y_tokenizer = y_tokenizer
        self.reverse_target_word_index = y_tokenizer.index_word
        self.reverse_source_word_index = x_tokenizer.index_word
        self.target_word_index = y_tokenizer.word_index
    
    def create_model(self):
        #encoder decoder LSTM architecture
        #encoder
        self.encoder_inputs = Input(shape=(self.max_code_len,), name='encoder_inputs')
        self.encoder_embedding = Embedding(self.x_vocab, self.embedding_dim, trainable = True)(self.encoder_inputs)
        encoder_lstm_1 = LSTM(self.latent_dim, return_sequences = True, return_state=True, dropout = 0.4, recurrent_dropout = 0.4, name='encoder_lstm_1')
        encoder_outputs_1, encoder_state_h_1, encoder_state_c_1 = encoder_lstm_1(self.encoder_embedding)
        encoder_lstm_2 = LSTM(self.latent_dim, return_sequences = True, return_state=True, dropout = 0.4, recurrent_dropout = 0.4, name='encoder_lstm_2')
        encoder_outputs_2, encoder_state_h_2, encoder_state_c_2 = encoder_lstm_2(encoder_outputs_1)
        encoder_lstm_3 = LSTM(self.latent_dim, return_sequences = True, return_state=True, dropout = 0.4, recurrent_dropout = 0.4, name='encoder_lstm_3')
        self.encoder_output, self.encoder_state_h, self.encoder_state_c = encoder_lstm_3(encoder_outputs_2)
        #decoder
        self.decoder_inputs = Input(shape=(None, ))
        self.decoder_embedding_layer = Embedding(self.y_vocab, self.embedding_dim, trainable = True)
        decoder_embedding = self.decoder_embedding_layer(self.decoder_inputs)
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences = True, return_state=True, dropout = 0.4, recurrent_dropout = 0.2, name='decoder_lstm')
        self.decoder_outputs, decoder_fwd_state, decoder_back_state = self.decoder_lstm(decoder_embedding, initial_state = [self.encoder_state_h, self.encoder_state_c])
        self.decoder_dense = TimeDistributed(Dense(self.y_vocab, activation='softmax'))
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        

    def plot_loss(self):
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        plt.savefig('model_loss.png')
    
    def train_model(self, x_train, y_train, x_val, y_val):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        #callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        checkpoint_path = 'cp.ckpt'
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True)
        callbacks = [early_stopping, checkpoint]
        #train model
        self.model.fit(
            [x_train, y_train[:, :-1]],
            y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:, 1:],
            epochs = 20,
            callbacks=callbacks,
            batch_size=128,
            validation_data=(
                [x_val, y_val[:, :-1]],
                y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]
            ),
        )
        self.model.save_weights('model_weights.h5')
        self.plot_loss()   
     

     #predictions
    def encode_decode(self):
        self.encoder_model = Model(
            inputs = self.encoder_inputs,
            outputs = [self.encoder_output, self.encoder_state_h, self.encoder_state_c]
        )
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_state_inputs = Input(shape = (self.max_code_len, self.latent_dim))

        decoder_embedding = self.decoder_embedding_layer(self.decoder_inputs)
        decoder_outputs, decoder_state_h, decoder_state_c = self.decoder_lstm(
            decoder_embedding,
            initial_state = [decoder_state_input_h, decoder_state_input_c]
        )
        decoder_outputs = self.decoder_dense(decoder_outputs)

        self.decoder_model = Model(
            [self.decoder_inputs] + [decoder_state_inputs, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs] + [decoder_state_h, decoder_state_c]
        )

    def decode_sequence(self,input_seq):
        enc_out, enc_h, enc_c = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.target_word_index['sostok']
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_token, h, c = self.decoder_model.predict([target_seq] + [enc_out, enc_h, enc_c])
            sampled_token_index = np.argmax(output_token[0, -1, :])
            sampled_token = self.reverse_target_word_index[sampled_token_index]
            if sampled_token != 'eostok':
                decoded_sentence += ' ' + sampled_token
            if sampled_token == 'eostok' or len(decoded_sentence.split() >= self.max_summary_len - 1):
                stop_condition = True
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            enc_out, enc_h, enc_c = h, c, enc_out
        return decoded_sentence
    
    def predict(self, input_string):
        input_seq = self.x_tokenizer.texts_to_sequences([input_string])
        input_seq = pad_sequences(input_seq, maxlen=self.max_code_len, padding='post')
        decoded_sentence = self.decode_sequence(input_seq.reshape(1, self.max_code_len, 1))
        return decoded_sentence
