import numpy as np
import math
from nltk.stem.wordnet import WordNetLemmatizer
import re

lemmatizer = WordNetLemmatizer()


def id2string(cap, tokenizer, batch=False):
    if batch:
        out = []
        for caption in cap:
            out.append(id2string(caption, tokenizer))
        return out
    else:
        filtered_cap = []
        for item in cap:
            if item == BOS_ID:
                filtered_cap = []
            elif item == EOS_ID:
                break
            else:
                filtered_cap.append(item)
        return tokenizer.decode(filtered_cap)


class CaptionReward:
    def __init__(self, dictionary, N):
        self.N = N
        self.dictionary = dictionary

    def cos_similarity(self, d1, d2):
        score = 0
        total1 = 0
        total2 = 0
        for k, v in d1.items():
            if k in d2:
                score += v * d2[k]
            total1 += v ** 2
        for k, v in d2.items():
            total2 += v ** 2
        return (score + 1e-8) / math.sqrt(total1 * total2 + 1e-8)

    def clean_text(self, text):
        text = re.sub('[.,;!?]', ' ', text)
        text = re.sub('[ ][ ]+', ' ', text).strip()
        text = text.lower()
        return text

    def tf_idf(self, text, order):
        text = self.clean_text(text)
        tf = dict()
        result = dict()
        count = 0
        tokens = [lemmatizer.lemmatize(item) for item in text.split(' ')]
        if len(tokens) < order:
            return {}
        for i in range(len(tokens) - order + 1):
            token = ' '.join(tokens[i:i + order])
            if token not in tf:
                tf[token] = 0
            tf[token] += 1
            count += 1
        for token, tf_value in tf.items():
            if token in self.dictionary[order]:
                df = self.dictionary[order][token]['df']
            else:
                df = 0
            result[token] = (tf[token] / count) * math.log((self.N + 1) / (df + 1))
        return result

    def cider(self, refs, c):
        final_score = 0
        for order in [1, 2, 3, 4]:
            ref_scores = []
            for ref in refs:
                ref_scores.append(self.tf_idf(ref, order=order))
            c_score = self.tf_idf(c, order=order)
            score = 0
            for ref_score in ref_scores:
                score += self.cos_similarity(c_score, ref_score)
            score /= len(refs)
            final_score += score
        return final_score / 4
