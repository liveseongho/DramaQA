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

def to_string(x, tokenizer):
    str = tokenizer.decode(x, skip_special_tokens=True)
    str = str.split()
    return str

def format_string(x, tokenizer, is_id = False):
    x = to_string(x, tokenizer)
    return {str(i): [v] for i, v in enumerate(x)}

# return socores between 0.0 ~ 1.0
def Bleu(output, target, tokenizer):
    pred_sentence = to_string(output, tokenizer)
    target_sentence = to_string(target, tokenizer)
    pred_sentence = [pred_sentence]

    score = sentence_bleu(pred_sentence, target_sentence)
    return score

# return socores between 0.0 ~ 1.0
def Meteor(output, target, tokenizer):
    pred_sentence = to_string(output, tokenizer)
    target_sentence = to_string(target, tokenizer)
    pred_sentence = [' '.join(pred_sentence)]
    target_sentence = ' '.join(target_sentence)

    score = meteor_score(pred_sentence, target_sentence)
    return score

# return socores between 0.0 ~ 1.0
# For Rouge, length of pred, target sentence must be equal.
def Rouge(output, target, tokenizer):
#    pred_sentence = format_string(output, tokenizer)
#    target_sentence = format_string(target, tokenizer)
#    print('pred_sentence : ', pred_sentence)
#    print('target_sentence : ', target_sentence)
#
#    scorer, method = rouge_scorers
#    score, scores = scorer.compute_score(pred_sentence, target_sentence)
#    return score

    return 0

# return socores between 1.0 ~ 10.0
def Cider(output, target, tokenizer):
#    pred_sentence = format_string(output, tokenizer)
#    target_sentence = format_string(target, tokenizer)
#
#    scorer, method = cider_scorers
#    score, scores = scorer.compute_score(pred_sentence, target_sentence)
#    return score

    return 0
