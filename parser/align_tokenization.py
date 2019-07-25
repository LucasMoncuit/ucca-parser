#!/usr/bin/env python3

import spacy

def tokenize_merge(file1, file2):
	nlp1 = spacy.load("en_core_web_sm")
	nlp2 = spacy.load("French model") #Find it
	with open(file1) as f1, open(file2) as f2, open("merged_tokenized", "w+") as ft:
		for line1, line2 in zip(f1,f2):
			doc1 = nlp1(line1)
			doc2 = nlp2(line2)
			for token in doc1:
				text_token = token.text
				ft.write(text_token + " ")
			ft.write("|||" + " ")
			for token in doc2:
				text_token = token.text
				ft.write(text_token + " ")
			ft.write("\n")

if __name__=='__main__':
	tokenize_merge("/home/lucas/PycharmProjects/ucca-parser-multi/europarl-v7.fr-en.en", "/home/lucas/PycharmProjects/ucca-parser-multi/europarl-v7.fr-en.fr")
