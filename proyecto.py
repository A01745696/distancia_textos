import nltk
#nltk.download("wordnet")
#nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer
from textblob import Word
from nltk.util import ngrams
from collections import Counter
import os
import tkinter as tk
from tkinter import filedialog
######################################################
message = []
#######################################################
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

def open_file():
    file_path = filedialog.askopenfilename(
        title="Select a Text File", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
            message.append(content)
            text_widget.delete(1.0, tk.END)  # Clear previous c ontent
            text_widget.insert(tk.END, "File Correct, This Window will close automatically! xD")
            root.after(3000,lambda:root.destroy())
############################################################

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

root = tk.Tk()
root.title("Mi Ventana")
text_widget = tk.Text(root, wrap="word", width=40, height=10)
text_widget.pack(pady=10)

open_button = tk.Button(root, text="Open File", command=open_file)
open_button.pack(pady=10)
root.mainloop()
mitexto = message[0]
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

##TRESHOLD##
unigramas_parecidos = [i for i in distancia_unigramas if i < 0.25]
bigramas_parecidos = [i for i in distancia_bigramas if i < 0.74]
trigramas_parecidos = [i for i in distancia_trigramas if i < 0.81]
titulo_textos = []
for x in unigramas_parecidos:
    titulo_textos.append("Unigrama: " + str(arr[distancia_unigramas.index(x)]))
for x in bigramas_parecidos:
    titulo_textos.append("Bigrama: " + str(arr[distancia_bigramas.index(x)]))
for x in trigramas_parecidos:
    titulo_textos.append("Trigrama: " + str(arr[distancia_trigramas.index(x)]))

print("Tu texto se parece a:")
for x in titulo_textos:
    print(x)
 


