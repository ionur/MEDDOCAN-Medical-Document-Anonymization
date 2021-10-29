import os
import pprint
import argparse
import numpy as np
import itertools
from glob import glob

import pickle
import nltk
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut

import spacy
nlp = spacy.load('es')

cess_sents = cess.tagged_sents()

uni_tag = ut(cess_sents)

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

parser.add_argument('--dataDir', type=str, required=True)
parser.add_argument('--train', dest="isTrain", action='store_true')
parser.add_argument('--dev', dest="isDev", action='store_true')
parser.add_argument('--test', dest="isTest", action='store_true')

__pos_tagger = PerceptronTagger()

def prepare(rawtext):
	# To process text in NLTK format
	sentences = nltk.sent_tokenize(rawtext.strip())
	sentences = [nltk.word_tokenize(sent) for sent in sentences]

	return sentences


def find_sent_nltk_format_v1(inputFile, ner_tags_ann_dict):


	list_of_words = []
	list_of_labels = []

	with open(inputFile, 'r') as f:
		rawtext = f.read()
		sentences = prepare(rawtext.strip())
		
		for sentence in sentences:
			for word in sentence:
				list_of_words.append(word)
				if word in ner_tags_ann_dict:
					list_of_possible_ner_tags = ner_tags_ann_dict[word]
					list_of_labels.append(list_of_possible_ner_tags[0])
					list_of_possible_ner_tags.pop(0)
					if len(list_of_possible_ner_tags) > 0:
						ner_tags_ann_dict[word] = list_of_possible_ner_tags
					else:
						del ner_tags_ann_dict[word]
				else:
					list_of_labels.append("O")
	return sentences
	

def process_annotation_file_v1(ann_file):
	ann_file_ner_dict = {}

	lines = []
	with open(ann_file, 'r') as f:
		lines = f.readlines()

	for line in sorted(lines, key=lambda line: int(line.split("\t")[1].split(" ")[1])):
		elements = line.split("\t")

		tagId = elements[0]
		tagName_startIdx_endIdx = elements[1]
		tagValue = elements[2].strip("\n")

		tagName_startIdx_endIdx_elements = tagName_startIdx_endIdx.split(" ")
		tagName = tagName_startIdx_endIdx_elements[0]
		startIdx = tagName_startIdx_endIdx_elements[1]
		endIdx = tagName_startIdx_endIdx_elements[2]

		words_in_tagValue = nltk.word_tokenize(tagValue)

		for idx, word in enumerate(words_in_tagValue):
			if idx == 0:
				value = "B-" + tagName
			else:
				value = "I-" + tagName

			if word in ann_file_ner_dict:
				old_value_list = ann_file_ner_dict[word]
				old_value_list.append(value)
				ann_file_ner_dict[word] = old_value_list
			else:
				ann_file_ner_dict[word] = [value]

	return ann_file_ner_dict


def process_annotation_file_v2(ann_file):
	lines = []
	with open(ann_file, 'r') as f:
		lines = f.readlines()

	list_of_annotation_tuples = []
	for line in sorted(lines, key=lambda line: int(line.split("\t")[1].split(" ")[1])):
		elements = line.split("\t")
		tagId = elements[0]
		tagName_startIdx_endIdx = elements[1]
		tagValue = elements[2].strip("\n")

		tagName_startIdx_endIdx_elements = tagName_startIdx_endIdx.split(" ")
		tagName = tagName_startIdx_endIdx_elements[0]
		startIdx = tagName_startIdx_endIdx_elements[1]
		endIdx = tagName_startIdx_endIdx_elements[2]

		ann_tuple = (startIdx, endIdx, tagName, tagValue)
		list_of_annotation_tuples.append(ann_tuple)


	return list_of_annotation_tuples

def create_ner_type_data_for_train(inputFile, ner_tags_ann_list):

	readSoFar = 0
	idx = 0

	outer_list = []
	
	with open(inputFile, 'r') as f:
		for line in f:	
			list_of_words = []
			list_of_labels = []
			sent_list = []
			lenOfCurrentLine = len(line)
			words_in_train_line = nltk.word_tokenize(line)
			index_of_words_in_train_line = []
			postag_of_words_in_train_line = []
			currIdx = 0

			if '``' in words_in_train_line:
				words_in_train_line = [w.replace('``', '"') for w in words_in_train_line]

			if "''" in words_in_train_line:
				words_in_train_line = [w.replace("''", '"') for w in words_in_train_line]

			for w in words_in_train_line:
				index_of_words_in_train_line.append(readSoFar + line.index(w, currIdx))
				currIdx += len(w)

			for w in words_in_train_line:
				tokens = nlp(w)
				for t in tokens:
					postag_of_words_in_train_line.append(t.pos_)

			
			found = 0
			if idx >= len(ner_tags_ann_list):
				for word__ in words_in_train_line:
					list_of_words.append(word__)
					list_of_labels.append("O")
				continue
					
			startIdx_from_ann_list = int(ner_tags_ann_list[idx][0])
			i = 0
			while startIdx_from_ann_list >= readSoFar and startIdx_from_ann_list < (readSoFar + lenOfCurrentLine):
				found = 1
				value = ner_tags_ann_list[idx][3]
				words_in_ann_line = nltk.word_tokenize(value)
				word1_ann = words_in_ann_line[0]

				flag = 1
				while i < len(words_in_train_line):
					if word1_ann == words_in_train_line[i]:
						j = 1
						while j < len(words_in_ann_line):
							ann_word = words_in_ann_line[j]
							if ann_word == words_in_train_line[i+1]:
								i = i + 1
								j = j + 1
							else:
								flag = 0
								break

						if flag != 0:
							tagName = ner_tags_ann_list[idx][2]
							for idx_, word in enumerate(words_in_ann_line):
								if idx_ == 0:
									list_of_words.append(word)
									list_of_labels.append("B-" + tagName)
								else:
									list_of_words.append(word)
									list_of_labels.append("I-" + tagName)
							i = i + 1
							break
						else:
							idx__ = len(list_of_labels)
							while idx__ < i:
								list_of_words.append(words_in_train_line[idx__])
								list_of_labels.append("O")
								idx__ = idx__ + 1
					else:
						list_of_words.append(words_in_train_line[i])
						list_of_labels.append("O")

					i = i + 1

				idx = idx + 1
				if idx >= len(ner_tags_ann_list):
					break
				else:
					startIdx_from_ann_list = int(ner_tags_ann_list[idx][0])

			if found == 0:
				for word__ in words_in_train_line:
					list_of_words.append(word__)
					list_of_labels.append("O")

			readSoFar = readSoFar + lenOfCurrentLine
			for word, label, startIndex, postag in zip(list_of_words, list_of_labels, index_of_words_in_train_line, postag_of_words_in_train_line):
				sent_list.append((word, label, startIndex, postag))
			outer_list.append(sent_list)
	return outer_list


# For test files (data without .ann files) Tags all words as 'None'
def create_ner_type_data_for_test(inputFile):
	outer_list = []
	with open(inputFile, 'r') as f:
		for line in f:	
			list_of_words = []
			list_of_labels = []
			sent_list = []
			
			words_in_train_line = nltk.word_tokenize(line)
		
			for word__ in words_in_train_line:
				list_of_words.append(word__)
				list_of_labels.append(None)

			for word, label in zip(list_of_words, list_of_labels):
				sent_list.append((word, label))
			outer_list.append(sent_list)

	return outer_list

def fname_to_docid(fname):
	docid = os.path.splitext(os.path.basename(fname))[0].split('.')[0]
	return docid


if __name__ == '__main__':
	args = parser.parse_args()
	pp.pprint(args)

	all_sentences = {}
	fnames = glob(args.dataDir + '/*.txt')

	if args.isTrain or args.isDev or args.isTest:
		for train_fname in fnames:
			print("Reading Training/Dev File = ", train_fname)
			annFileName = os.path.basename(train_fname).replace('.txt', '.ann')
			annFileNameFullPath = os.path.join(args.dataDir, annFileName)
			print("Reading Annotation file = ", annFileNameFullPath)
			ner_tags_ann_list = process_annotation_file_v2(annFileNameFullPath)
			list_of_sentences = create_ner_type_data_for_train(train_fname, ner_tags_ann_list)
			docid = fname_to_docid(train_fname)
			all_sentences[docid] = list_of_sentences
	else:
		for test_fname in fnames:
			print("Reading Test file = ", test_fname)
			list_of_sentences = create_ner_type_data_for_test(test_fname)
			docid = fname_to_docid(test_fname)
			all_sentences[docid] = list_of_sentences

	pickle_filename = ""
	if args.isTrain:
		pickle_filename = "./train_word_ner_startidx_dict.pickle"
	elif args.isDev:
		pickle_filename = "./dev_word_ner_startidx_dict.pickle"
	elif args.isTest:
		pickle_filename = "./test_word_ner_startidx_dict.pickle"

	with open(pickle_filename, 'wb') as handle:
		pickle.dump(all_sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
