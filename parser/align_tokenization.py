#!/usr/bin/env python3

import spacy

def tokenize_merge(file1, file2):
	nlp1 = spacy.load("en_core_web_sm")
	nlp2 = spacy.load("fr_core_news_sm")
	with open(file1) as f1, open(file2) as f2, open("merged_tokenized", "w+") as ft:
		for line1, line2 in zip(f1,f2):
			doc1 = nlp1(line1)
			doc2 = nlp2(line2)
			tokenized_sentence = []
			for token1 in doc1:
				if token1.text != "\n":
					tokenized_sentence.append(token1.text)
			tokenized_sentence += ["|||"]
			for token2 in doc2:
				tokenized_sentence.append(token2.text)
			for token in tokenized_sentence:
				if token != "\n":
					ft.write(token + " ")
				else:
					ft.write(token)
			del tokenized_sentence

if __name__=='__main__':
	tokenize_merge("/home/lucas/PycharmProjects/ucca-parser-multi/europarl-v7.fr-en.en", "/home/lucas/PycharmProjects/ucca-parser-multi/europarl-v7.fr-en.fr")
