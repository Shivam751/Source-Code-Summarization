def bleu_score(ref, pred):
    if(pred == ''):
        return 0
    words = pred.split(' ')
    words = [word for word in words if word != '']
    vocab = set(words)
    ref_words = ref.split(' ')
    ref_words = [word for word in ref_words if word != '']
    num = 0
    den = 0
    for word in vocab:
        den += words.count(word)
        num += ref_words.count(word)
    return num/den