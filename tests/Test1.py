#TEST USING FID-013

import nltk
#nltk.download("wordnet")
#nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer
from textblob import Word
from nltk.util import ngrams
from collections import Counter
import os

def limpieza(parrafo):
    parrafo = parrafo.replace(".","")
    parrafo = parrafo.replace(",","")
    parrafo = parrafo.replace("\n","")
    parrafo = parrafo.lower()
    return parrafo

def lemmatizar(l):
  #lemme = WordNetLemmatizer()
  lnuevo = []
  for word in l:
    lnuevo.append(Word(word).lemmatize())
    #lnuevo.append(lemme.lemmatize(word))
  return lnuevo

def ngrama(l,n):
  grama = ngrams(l,n)
  gramalista = list(grama)
  gramalista = [' '.join(tup) for tup in gramalista]
  return gramalista

# sacas los diccionarios (vectores) de los ngramas para poder sacar la distancia despues
def cosine_distance(ngrams1, ngrams2):
    vec1 = Counter(ngrams1)
    vec2 = Counter(ngrams2)
    all_ngrams = set(vec1.keys()).union(set(vec2.keys()))
    dot_product = sum(vec1[ngram] * vec2[ngram] for ngram in all_ngrams)
    norm1 = sum(vec1[ngram] ** 2 for ngram in all_ngrams) ** 0.5
    norm2 = sum(vec2[ngram] ** 2 for ngram in all_ngrams) ** 0.5
    return 1 - dot_product / (norm1 * norm2)

arr = os.listdir()
arr = os.listdir('./db')
textos = []
for texto in arr:
    with open("./db/"+str(texto), encoding='utf8') as f:
        contents = f.read()
        textos.append(limpieza(contents))

unigramas = []
bigramas = []
trigramas = []
for texto in textos:
    l = texto.split(" ")
    l = lemmatizar(l)
    unigramas.append(ngrama(l,1))
    bigramas.append(ngrama(l,2))
    trigramas.append(ngrama(l,3))

mitexto = "Source-code Similarity Detection and Detection Tools Used in Academia: A Systematic Review Teachers deal with plagiarism on a regular basis, so they try to prevent and detect plagiarism, a task that is complicated by the large size of some classes. Students who cheat often try to hide their plagiarism (obfuscate), and many different similarity detection engines (often called plagiarism detection tools) have been built to help teachers. This article focuses only on plagiarism detection and presents a detailed systematic review of the field of source-code plagiarism detection in academia. This review gives an overview of definitions of plagiarism, plagiarism detection tools, comparison metrics, obfuscation methods, datasets used for comparison, and algorithm types. More insidiously, because of its non-deterministic approach, MOSSAD can, from a single program, generate dozens of variants, which are classified as no more suspicious than legitimate assignments. A detailed study of MOSSAD across a corpus of real student assignments demonstrates its efficacy at evading detection. A user study shows that graduate student assistants consistently rate MOSSAD-generated code as just as readable as authentic student code. This work motivates the need for both research on more robust plagiarism detection tools and greater integration of naturally plagiarism-resistant methodologies like code review into computer science education."
mitexto = limpieza(mitexto)
mitexto = mitexto.split(" ")
unigrama = ngrama(mitexto,1)
bigrama = ngrama(mitexto,2)
trigrama = ngrama(mitexto,3)

distancia_unigramas = []
distancia_bigramas = []
distancia_trigramas = []

for i in range(len(unigramas)):
    distancia_unigramas.append(cosine_distance(unigramas[i], unigrama))
    distancia_bigramas.append(cosine_distance(bigramas[i], bigrama))
    distancia_trigramas.append(cosine_distance(trigramas[i], trigrama))

print(distancia_unigramas)
print(distancia_bigramas)
print(distancia_trigramas)
