#TEST USING FID-010

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

mitexto = "Artificial intelligence was one of the emerging technologies that simulated human intelligence in machines by programming them to think like human beings and mimic their actions. An autonomous vehicle could function by itself and carry out necessary functions without any human involvement. This innovative technology provided increased passenger safety, less congested roads, congestion reduction, optimum traffic flow, lower fuel consumption, less pollution, and better travel experiences. Autonomous vehicles played a vital role in industry, agriculture, transportation, and military applications. The autonomous vehicle's activities were supported by sensor data and a few artificial intelligence systems. Artificial intelligence was the collection of data, path planning, and execution in autonomous vehicles that required some machine learning techniques that were a part of artificial intelligence. But this came with some privacy issues and security concerns. Security was an important concern for autonomous vehicles. The issues of cybersecurity while incorporating artificial intelligence in autonomous vehicles were covered in this article, along with the growing technology of self-driving automobiles."
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
