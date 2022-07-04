import spacy
import re
import numpy as np
import pandas as pd

def clean_text(column):
    for row in column:
        row = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1',  str(row))).split()
        row = ' '.join(row)
        row = re.sub("(\\t)", " ", str(row)).lower()
        row = re.sub("(\\r)", " ", str(row)).lower()
        row = re.sub("(\\n)", " ", str(row)).lower()

        # Remove the characters - <>()|&©ø"',;?~*!
        row = re.sub(r"[<>()|&©ø\[\]\'\",.\}`$\{;@?~*!+=_\//1234567890]", " ", str(row)).lower()
        row = re.sub(r"\\b(\\w+)(?:\\W+\\1\\b)+", "", str(row)).lower()

        # Remove punctuations at the end of a word
        row = re.sub("(\.\s+)", " ", str(row)).lower()
        row = re.sub("(\-\s+)", " ", str(row)).lower()
        row = re.sub("(\:\s+)", " ", str(row)).lower()

        # Replace any url to only the domain name
        try:
            url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
            repl_url = url.group(3)
            row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
        except:
            pass

        # Remove multiple spaces
        row = re.sub("(\s+)", " ", str(row)).lower()

        # Remove the single character hanging between any two spaces
        row = re.sub("(\s+.\s+)", " ", str(row)).lower()

        yield row


def preprocess_data(data):
    cleaned_code = clean_text(data['code'])
    cleaned_summary = clean_text(data['summary'])

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    code = [str(doc) for doc in nlp.pipe(cleaned_code, batch_size=50)]
    summary = [ str(doc)  for doc in nlp.pipe(cleaned_summary, batch_size=50)]

    data['code'] = code
    data['summary'] = summary

def shorten_data(data):
    max_code_len = 65 #0.98 qunatile 
    max_summary_len = 13 #0.98 qunatile

    # Select the Summaries and Text which fall below max length 
    cleaned_code = np.array(data['code'])
    cleaned_summary= np.array(data['summary'])

    short_text = []
    short_summary = []

    for i in range(len(cleaned_code)):
        if len(cleaned_summary[i].split()) <= max_summary_len and len(cleaned_code[i].split()) <= max_code_len:
            short_text.append(cleaned_code[i])
            short_summary.append(cleaned_summary[i])
            
    short_df = pd.DataFrame({'code': short_text,'summary': short_summary})
    short_df['summary'] = short_df['summary'].apply(lambda x: 'sostok ' + x + ' eostok')
    
    return short_df.dropna()



