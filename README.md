# Source-Code-Summarization

## Task: Metadata generation from Code using Sequence to Sequence Translation Model

# Dataset used: 
[Code Search Net](https://drive.google.com/file/d/1kR4Ml3IvJVXtx-VUZXyWHGAEN5Z9w0j_/view?usp=sharing)

Source-Code-Summarizer is a sequence to sequence translation architechture to generate docstring from code snippets (Python programming language).

# Apporach
The code and docstring pairs will be processed and passed to a encoder-decoder model to generate doctrings.

#Steps
1. Text Preprocessing
2. Text Tokenization
3. Sequence Padding
4. Embedding + Encoder
5. Embedding + Decoder
6. Generation
7. Evaluation: BLEU Score(1-gram)

# Model
1. *Encoder*
Embedding(code_vocab, embedding_dim) -> LSTM(latent_dim) -> LSTM(latent_dim) -> LSTM(latent_dim)
2. *Decoder*
Embedding(summary_vocab, embedding_dim) -> LSTM(latent_dim)
3. *Output*
TimeDistributed(Dense(summary_vocab)) [softmax activation]

*loss*: sparse_categorical_crossentropy
*optimizer*: rms_prop/adam

# Trained Model
The model has 11M parameters and trained for about 15hrs on GPU. The trained model can be downloaded from [here](https://drive.google.com/file/d/1dT_hppZlqadezWjqIpPnrpCMJ28r0Kyz/view?usp=sharing)


