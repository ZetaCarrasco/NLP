text="This paper further discusses the RAG training, including RAGwith/without datastore update. Then, we introduce the application of RAG in representative natural language processing tasks and industrial scenarios. Finally, this paper discusses the future directions and challenges of RAG for promoting its development"
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
print ('---')
print(text)
print("lengthText:", len(text))
print('---')
print(tokens)
print("lengthTokens:", len(tokens))

def get_stats(ids):
    counts ={}
    for pair in zip(ids, ids[1:]):
        counts[pair]= counts.get(pair,0)+1
        return counts
stats=get_stats(tokens)
print(stats)
#print(sorted((V,K)for K,V in stats.items()), reverse=True())        

top_pair= max(stats, key=stats.get)
top_pair


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i< len(ids):
        if i <len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids

print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))

tokens2 = merge(tokens, top_pair, 256)
print(tokens2)
print('length:', len(tokens2))


vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
print(f"merging {pair} into a new token {idx}")
ids = merge(ids, pair, idx)
merges [pair] = idx    


print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):2f}X")


#decoding
vocab = {idx: bytes ([idx] )for idx in range(256) }
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab [p0] + vocab [p1]

def decode (ids): 
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text
print(decode([33]))


#encoding
def encode(text):
    tokens =  list(text.encode("utf-8"))
    while True:
        stats =  get_stats(tokens)
        pair = min(stats, key= lambda p: merges.get(p, float ("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens =  merge(tokens, pair, idx)
    return tokens
print(encode("Have a nice day!")) 

#para verificar se o decode e encode está correto
text1 = decode(encode(text))
print(text1 == text)
#verificando outro texto que o tokenizer não conhece
valtext = "This acceleration was largely driven by three synergistic trends. First, large amounts of spoken and written material became widely available through the auspices of the Linguistic Data Consortium (LDC)."
valtext2 = decode(encode(valtext))
print(valtext2 == valtext)


#divide o texto em uma lista de texto e cada um é processado diferente
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print(re.findall(gpt2pat, "How is going. I'm great"))  


#trabalha com os espaços 
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.encode("       hello world!!!!"))

enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode("       hello world!!!!"))




import os, json

with open('encoder.json', 'r') as f:
    encoder = json.load(f)
    
with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
print(len(encoder))