import torch

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

scorers = {
    'rouge': (Rouge(), "ROUGE_L"),
    'cider': (Cider(), "CIDEr")
}
metrics = ['rouge', 'cider']
scorers = [v for k, v in scorers.items() if k in metrics]
rouge_scorers = scorers[0]
cider_scorers = scorers[1]

def to_string(x, vocab):
    str = []
    for i in range(x.shape[0]):
        str.append(vocab.itos[x[i].tolist()])
    return str

def format_string(x, vocab):
    x = to_string(x, vocab)
    return {str(i): [v] for i, v in enumerate(x)}

# return socores between 0.0 ~ 1.0
def Bleu(output, target, vocab):
    pred_sentence = to_string(output, vocab)
    target_sentence = to_string(target, vocab)
    pred_sentence = [pred_sentence]

    score = sentence_bleu(pred_sentence, target_sentence)
    return score

# return socores between 0.0 ~ 1.0
def Meteor(output, target, vocab):
    pred_sentence = to_string(output, vocab)
    target_sentence = to_string(target, vocab)
    pred_sentence = [' '.join(pred_sentence)]
    target_sentence = ' '.join(target_sentence)

    score = meteor_score(pred_sentence, target_sentence)
    return score

# return socores between 0.0 ~ 1.0
def Rouge(output, target, vocab):
    pred_sentence = format_string(output, vocab)
    target_sentence = format_string(target, vocab)

    scorer, method = rouge_scorers
    score, scores = scorer.compute_score(pred_sentence, target_sentence)
    return score

# return socores between 1.0 ~ 10.0
def Cider(output, target, vocab):
    pred_sentence = format_string(output, vocab)
    target_sentence = format_string(target, vocab)

    scorer, method = cider_scorers
    score, scores = scorer.compute_score(pred_sentence, target_sentence)
    return score

