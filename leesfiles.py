import PyPDF2 as pp
import os
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import spacy
import nl_core_news_sm
import pandas as pd


punctuations = "()-;:,.?!'\""

testing = False
if testing: # We do not go through all files but only a selection
	filenames = ['1997_Rozema-van der Veen, Frieda_Als het donker wordt.pdf']
else:
	filenames = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]

nlp = nl_core_news_sm.load()

for file in filenames:
    with open(file,'rb') as pdfFileObj:
        pdfReader = pp.PdfFileReader(pdfFileObj) 
        tokenized_word = []
        for page in range(pdfReader.getNumPages()):
            pageObj = pdfReader.getPage(page) 
            tokenized_word+=[word.lower() for word in word_tokenize(pageObj.extractText()) if word and not word in punctuations] 
        doc = nlp(" ".join(tokenized_word))
        spacy_pos_tagged = [word.text for word in doc if word.pos_ == 'NOUN']
        plt.ion()
        fig = plt.figure(figsize = (10,4))
        plt.gcf().subplots_adjust(bottom=0.15) # to avoid x-ticks cut-off
        fdist = FreqDist(spacy_pos_tagged)
        fdist.plot(30,cumulative=False)
        fig.savefig(file[:-4]+'_freq.png', bbox_inches = "tight")
        plt.ioff()
