import torch
import transformers
from transformers import BertModel,BertConfig,BertForMaskedLM
from transformers import BertTokenizer
import preprocess

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
configuration = BertConfig(num_hidden_layers=4,max_position_embeddings=64,hidden_size=2400)
model = BertModel(configuration)
tokenizer = BertTokenizer.from_pretrained("ngram_3.txt")

text = "[CLS] RDFTHTIIDNSDLFSESRNTRLG [SEP]"

print(tokenizer.tokenize(text))

