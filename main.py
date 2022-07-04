import pandas as pd
import numpy as np
from preprocess_data import preprocess_data, shorten_data
from sklearn.model_selection import train_test_split
from tokenize_text import tokenize_data
from Code_Summarization import code2summary
import re

def clean(code):
    code = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1',  str(code))).split()
    code = ' '.join(code)
    code = re.sub("(\\t)", " ", str(code)).lower()
    code = re.sub("(\\r)", " ", str(code)).lower()
    code = re.sub("(\\n)", " ", str(code)).lower()

    # Remove the characters - <>()|&©ø"',;?~*!
    code = re.sub(r"[<>()|&©ø\[\]\'\",.\}`$\{;@?~*!+=_\//1234567890]", " ", str(code)).lower()
    code = re.sub(r"\\b(\\w+)(?:\\W+\\1\\b)+", "", str(code)).lower()

    # Remove punctuations at the end of a word
    code = re.sub("(\.\s+)", " ", str(code)).lower()
    code = re.sub("(\-\s+)", " ", str(code)).lower()
    code = re.sub("(\:\s+)", " ", str(code)).lower()

    # Replace any url to only the domain name
    try:
        url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(code))
        repl_url = url.group(3)
        code = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(code))
    except:
        pass

    # Remove multiple spaces
    code = re.sub("(\s+)", " ", str(code)).lower()

    # Remove the single character hanging between any two spaces
    code = re.sub("(\s+.\s+)", " ", str(code)).lower()
    return code


def main():
    path = 'python_dataset.csv'
    data = pd.read_csv(path)
    preprocess_data(data)
    data_post_processing = shorten_data(data)
    x_train, x_val, y_train, y_val = train_test_split(
        np.array(data_post_processing['code']),
        np.array(data_post_processing['summary']),
        test_size=0.15,
        random_state=0,
        shuffle=True)
    max_code_len = 65
    max_summary_len = 13
    x_train, y_train, x_val, y_val, x_tokenizer, y_tokenizer = tokenize_data(
        x_train, x_val, y_train, y_val, max_code_len=max_code_len, max_summary_len=max_summary_len)
    x_vocab = x_tokenizer.num_words + 1
    y_vocab = y_tokenizer.num_words + 1
    latent_dim = 300
    embedding_dim = 200

    code_summarizer = code2summary(latent_dim, embedding_dim, max_code_len, max_summary_len, x_vocab, y_vocab, x_tokenizer, y_tokenizer)
    code_summarizer.create_model()
    code_summarizer.train_model(x_train, y_train, x_val, y_val)

    #prediction
    code_summarizer.encode_decode()
    code = "sum = a + b"
    #clean code
    code = clean(code)
    pred = code_summarizer.predict(code)
    print(pred)


if __name__ == "__main__":
    main()





