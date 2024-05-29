#TEST USING FID-005

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

mitexto = "Measures of success for facial feminization surgery (FFS) have previously included improved rates of external gender perception as female and patient-reported outcome measures.  Emotion recognition is the ability to precisely infer human emotions from numerous sources and modalities using questionnaires, physical signals, and physiological signals. In this study, we used artificial intelligence facial recognition software to objectively evaluate the effects of FFS on both perceived gender and age among male-to-female transgender patients, as well as their relationship with patient facial satisfaction. FFS was associated with a decrease in perceived age relative to the patient’s true age (−2.4 y, P<0.001), with older patients experiencing greater reductions. Pearson correlation matrix found no significant relationship between improved female gender typing and patient facial satisfaction. Transfeminine patients experienced improvements in satisfaction with facial appearance, perceived gender, and decreases in perceived age following FFS. Notably, patient satisfaction was not directly associated with improved AI-gender typing, suggesting that other factors may influence patient satisfaction. After a thorough analysis and discussion, we selected 142 journal articles using PRISMA guidelines. The review provides a detailed analysis of existing studies and available datasets of emotion recognition. Our review analysis also presented potential challenges in the existing literature and directions for future research."
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
