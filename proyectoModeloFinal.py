import tkinter as tk
from tkinter import filedialog
import os
import heapq
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
import spacy

##############_TEXTOS_#########################
textos1 = []
name1 = []
textos2 = []
name2 = []
#############_FUNCIONES_DE_VENTANA_####################

#Funcion de leer los textos individuales
def textovstexto():
  file1 = filedialog.askopenfilename(
     title="Select a Text File", filetypes=[("Text files", ".txt")])
  file2 = filedialog.askopenfilename(
     title="Select a Text File", filetypes=[("Text files", ".txt")])  
  text_widget.delete(1.0, tk.END)

  if file1 and file2:
    text_widget.insert(tk.END,"Los siguientes textos se compararan por cada tipo de plagio (Esta ventana se cerrará sola xD)\n")
    text_widget.insert(tk.END,file1)
    with open(file1, 'r', encoding='utf8') as file:
      content = file.read()
      textos1.append(content)
    text_widget.insert(tk.END,"\n------VS------\n")
    text_widget.insert(tk.END,file2)
    with open(file2, 'r', encoding='utf8') as file:
      content = file.read()
      textos2.append(content)
    root.after(5000, lambda:root.destroy())

  else:
     text_widget.insert(tk.END,"Favor de escoger los 2 documentos de forma individual\n")

#Funcion de leer todos los valores de la carpeta correspondiente y el texto individual
def textovsfolder():
  file1 = filedialog.askopenfilename(
    title="Select a Text File", filetypes=[("Text files", ".txt")])
  folder_path = filedialog.askdirectory()  # Open a folder selection dialog
  
  if folder_path and file1:
    with open(file1, 'r', encoding='utf8') as file:
      content = file.read()
      textos1.append(content)
    text_widget.insert(tk.END,"Los siguientes textos se compararan para cada tipo de plagio (Esta ventana se cerrará sola xD)\n")
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END,file1)
    text_widget.insert(tk.END,"\n------VS------\n")
    for item in os.listdir(folder_path):
      if ".txt" in item:
        name2.append(item)
        with open(folder_path+'/'+item, 'r', encoding='utf8') as file:
          content = file.read()
          textos2.append(content)
        text_widget.insert(tk.END, str(item + "\n"))  # Insert folder contents into Listbox
      root.after(5000, lambda:root.destroy())

  else:
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END,"Favor de escoger el archivo y la carpeta de forma correcta\n")

#Funcion de leer todos los valores de las carpeta correspondiente
def foldervsfolder():
  text_widget.delete(1.0, tk.END)
  text_widget.insert(tk.END,"Los siguientes textos se compararan para cada tipo de plagio (Esta ventana se cerrará sola xD)\n")
  folder_path1 = filedialog.askdirectory()  # Open a folder selection dialog
  folder_path2 = filedialog.askdirectory()  # Open a folder selection dialog

  if folder_path1 and folder_path2:
    for item in os.listdir(folder_path1):
      if ".txt" in item:
        name1.append(item)
        with open(folder_path1+'/'+item, 'r', encoding='utf8') as file:
          content = file.read()
          textos1.append(content)
        text_widget.insert(tk.END, str(item + "\n"))  # Insert folder contents into Listbox
    text_widget.insert(tk.END,"\n------VS------\n")
    for item in os.listdir(folder_path2):
      if ".txt" in item:
        name2.append(item)
        with open(folder_path2+'/'+item, 'r', encoding='utf8') as file:
          content = file.read()
          textos2.append(content)
        text_widget.insert(tk.END, str(item + "\n"))  # Insert folder contents into Listbox
    root.after(5000, lambda:root.destroy())

  else:
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END,"Favor de escoger las carpetas\n")

##########_FUNCIONES_DE_MODELO_BERT_##################
# Descargar el recurso 'punkt' de NLTK
#nltk.download('punkt')

# Carga el tokenizador y el modelo BERT preentrenado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_sm")

def get_bert(sentence, model, tokenizer):
  inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
  outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Limpieza del texto
def limpieza(parrafo):
  parrafo = parrafo.replace("\n","")
  return parrafo

def get_bert_embedding(sentence):
  inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
  with torch.no_grad():
      outputs = model(**inputs)
  return torch.mean(outputs.last_hidden_state, dim=1)

def detect_insertion_or_replacement(text1, text2):
  sentences1 = sent_tokenize(text1)
  sentences2 = sent_tokenize(text2)
  embeddings1 = [get_bert(sent, model, tokenizer) for sent in sentences1]
  embeddings2 = [get_bert(sent, model, tokenizer) for sent in sentences2]
  similarities = []

  for embedding1 in embeddings1:
    max_sim = 0
    for embedding2 in embeddings2:
        similarity = cosine_similarity(embedding1, embedding2)
        max_sim = max(max_sim, similarity.item())
    similarities.append(max_sim)
    
    average = np.mean(similarities)
  return average

def detect_reordering(text1, text2, beam_width=3):
  sentences1 = sent_tokenize(text1)
  sentences2 = sent_tokenize(text2)
  sentences2_embeddings = [get_bert(sent, model, tokenizer) for sent in sentences2]
  
  def calculate_similarity(perm):
      avg_sim = 0
      for sent1, sent2_embedding in zip(sentences1, perm):
          embedding1 = get_bert(sent1, model, tokenizer)
          similarity = cosine_similarity(embedding1, sent2_embedding)
          avg_sim += similarity.item()
      avg_sim /= len(sentences1)
      return avg_sim

  # Inicialización del haz con las oraciones originales y su similitud
  beam = [([], 0)]  # Cada elemento es una tupla (perm, sim)

  for _ in range(len(sentences1)):
    new_beam = []
    for perm_indices, sim in beam:
      for i, sent2_embedding in enumerate(sentences2_embeddings):
        if i not in perm_indices:
          new_perm_indices = perm_indices + [i]
          new_perm_embeddings = [sentences2_embeddings[j] for j in new_perm_indices]
          new_sim = calculate_similarity(new_perm_embeddings)
          heapq.heappush(new_beam, (new_perm_indices, new_sim))
          if len(new_beam) > beam_width:
              heapq.heappop(new_beam)
    if not new_beam:# Asegurarse de que siempre haya elementos en el haz
      new_beam = beam
    beam = new_beam

  best_perm_indices, max_avg_sim = max(beam, key=lambda x: x[1])
  best_perm_sentences = [sentences2[i] for i in best_perm_indices]
  
  return max_avg_sim, best_perm_sentences


def detect_grammar_changes(text1, text2):
  lemmatizer = nlp.get_pipe("lemmatizer")
  lemmatizer.mode
  doc1 = nlp(text1)
  doc2 = nlp(text2)
  lematizado1 = ' '.join([token.lemma_ for token in doc1])
  lematizado2 = ' '.join([token.lemma_ for token in doc2])
  embedding1 = get_bert_embedding(lematizado1)
  embedding2 = get_bert_embedding(lematizado2)
  similarity = cosine_similarity(embedding1, embedding2)
  return similarity.item()

def detect_paraphrasing(text1, text2):
  embedding1 = get_bert_embedding(text1)
  embedding2 = get_bert_embedding(text2)
  similarity = cosine_similarity(embedding1, embedding2)
  return similarity.item()

def detect_plagiarism(text1, text2):
  results = {}
  # Detectar parafraseo
  paraphrase_similarity = detect_paraphrasing(text1, text2)
  results["paraphrasing"] = paraphrase_similarity
  # Detectar cambios gramaticales (tiempo y voz)
  changes = detect_grammar_changes(text1, text2)
  results["grammar_changes"] = changes
  # Detectar inserción o reemplazo de frases
  similarities = detect_insertion_or_replacement(text1, text2)
  results["insertion_or_replacement"] = similarities
  # Detectar desorden de frases
  max_sim, best_perm = detect_reordering(text1, text2)
  results["reordering"] = max_sim
  return results

##################_MAIN_###############################
# Create the main window
root = tk.Tk()
root.title("File Explorer")
# Create a button to browse for a folder
onevsone_button = tk.Button(root, text="Texto vs Texto", command=textovstexto)
onevsone_button.pack(pady=10)
onevsfolder_button = tk.Button(root, text="Texto vs Folder", command=textovsfolder)
onevsfolder_button.pack(pady=10)
foldervsfolder_button = tk.Button(root, text="Folder vs Folder", command=foldervsfolder)
foldervsfolder_button.pack(pady=10)
#texto
text_widget = tk.Text(root, wrap="word", width=60, height=10)
text_widget.pack(pady=10)
# Start the Tkinter main loop
root.mainloop()

if len(textos1) == 1 and len(textos2) == 1:
  text1 = limpieza(textos1[0])
  text2 = limpieza(textos2[0])
  results = detect_plagiarism(text1, text2)

  if float(results["insertion_or_replacement"]) > .90:
    print("insertion_or_replacement: "+ str(results["insertion_or_replacement"]))
  if float(results["reordering"] > .90):
    print("reordering: "+ str(results["reordering"]))
  if float(results["grammar_changes"]) > .92:
    print("grammar_changes: "+ str(results["grammar_changes"]))
  if float(results["paraphrasing"] > .99):
    print("grammar_changes: "+ str(results["grammar_changes"]))

#  print("insertion_or_replacement: "+ str(results["insertion_or_replacement"]))
#  print("reordering: "+ str(results["reordering"]))
#  print("grammar_changes: "+ str(results["grammar_changes"]))
#  print("paraphrasing: "+ str(results["paraphrasing"]))

if len(textos1) == 1 and len(textos2) >= 2:
  text1 = limpieza(textos1[0])
  textfolder2 = []
  results = []
  i = 0
  for text in textos2:
    text2 = limpieza(text)
    results.append([name2[i], detect_plagiarism(text1, text2)])
    i+=1
  for result in results:
    if float(results["insertion_or_replacement"]) > .90:
      print("insertion_or_replacement: "+ str(results["insertion_or_replacement"]))
    if float(results["reordering"] > .90):
      print("reordering: "+ str(results["reordering"]))
    if float(results["grammar_changes"]) > .92:
      print("grammar_changes: "+ str(results["grammar_changes"]))
    if float(results["paraphrasing"] > .99):
      print("grammar_changes: "+ str(results["grammar_changes"]))
    else:
      print("There is not plagiarism")

if len(textos1) >= 2 and len(textos2) >= 2:
  results = []
  i = 0
  j = 0
  for text in textos1:
    text1 = limpieza(text)
    for text2 in textos2:
      text2 = limpieza(text2)
      results.append([name1[i], [name2[j], detect_plagiarism(text1, text2)]])
      j+=1
    i+=1
    j=0
  for result in results:
    if float(results["insertion_or_replacement"]) > .90:
      print("insertion_or_replacement: "+ str(results["insertion_or_replacement"]))
    if float(results["reordering"] > .90):
      print("reordering: "+ str(results["reordering"]))
    if float(results["grammar_changes"]) > .92:
      print("grammar_changes: "+ str(results["grammar_changes"]))
    if float(results["paraphrasing"] > .99):
      print("grammar_changes: "+ str(results["grammar_changes"]))
    else:
      print("There is not plagiarism")