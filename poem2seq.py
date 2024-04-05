from jiayan import load_lm
from jiayan import CharHMMTokenizer
import jieba
text = '床前明月光，疑似地上霜。举头望明月，低头思故乡。'
lm = load_lm('CRF_model/jiayan.klm')
tokenizer = CharHMMTokenizer(lm)
print(list(tokenizer.tokenize(text)))
print(list(jieba.cut(text, use_paddle=True)))