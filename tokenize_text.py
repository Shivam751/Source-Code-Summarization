from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

def tokenize_data(x_train, x_val, y_rain, y_val, max_code_len, max_summary_len):
    code_tokenizer = Tokenizer()
    code_tokenizer.fit_on_texts(list(x_train))
    thresh = 5
    cnt_infrequent = 0
    total_cnt = 0
    for word, cnt in code_tokenizer.word_counts.items():
        total_cnt += 1
        if cnt < thresh:
            cnt_infrequent += 1

    x_tokenizer = Tokenizer(num_words=total_cnt - cnt_infrequent)
    x_tokenizer.fit_on_texts(list(x_train))
    x_train = x_tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen=max_code_len, padding='post')
    x_val = x_tokenizer.texts_to_sequences(x_val)
    x_val = pad_sequences(x_val, maxlen=max_code_len, padding='post')

    summary_tokenizer = Tokenizer()
    summary_tokenizer.fit_on_texts(list(y_rain))
    thresh = 5
    cnt_infrequent = 0
    total_cnt = 0
    for word, cnt in summary_tokenizer.word_counts.items():
        total_cnt += 1
        if cnt < thresh:
            cnt_infrequent += 1
    
    y_tokenizer = Tokenizer(num_words=total_cnt - cnt_infrequent)
    y_tokenizer.fit_on_texts(list(y_rain))
    y_train = y_tokenizer.texts_to_sequences(y_train)
    y_train = pad_sequences(y_train, maxlen=max_summary_len, padding='post')
    y_val = y_tokenizer.texts_to_sequences(y_val)
    y_val = pad_sequences(y_val, maxlen=max_summary_len, padding='post')

    #dump x_tokenizer and y_tokenizer
    with open('x_tokenizer.pickle', 'wb') as handle:
        pickle.dump(x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('y_tokenizer.pickle', 'wb') as handle:
        pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return x_train, y_train, x_val, y_val, x_tokenizer, y_tokenizer
