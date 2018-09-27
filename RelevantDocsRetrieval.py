# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:05:39 2018

@author: mayuri
"""

from nltk.tokenize import WordPunctTokenizer
import math
import glob
import sys
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from html.parser import HTMLParser
from html.entities import name2codepoint

"""
Custom HTMLParser class to parse text in SGML format
"""
class MyHTMLParser(HTMLParser):
    
    def __init__(self):
        HTMLParser.__init__(self)
        self.titleTag = False
        self.textTag = False
        self.docnoTag = False
        self.lasttag = None
        self.title = None
        self.text = None
        self.docno = None

    def handle_starttag(self, tag, attrs):
        #print("Start tag:", tag)
        if tag == 'docno':
            self.docnoTag = True
            self.lasttag = tag
                
        elif tag == 'title':
            self.titleTag = True
            self.lasttag = tag
            
        elif tag == 'text':
            self.textTag = True
            self.lasttag = tag

    def handle_endtag(self, tag):
        #print("End tag  :", tag)
        if tag == 'docno':
            self.docnoTag = False
                
        elif tag == 'title':
            self.titleTag = False
            
        elif tag == 'text':
            self.textTag = False

    def handle_data(self, data):

        if self.lasttag == 'docno' and self.docnoTag:
            self.docno = data
            #print('data docno:', data)
        elif self.lasttag == 'title' and self.titleTag:
            self.title = data
            #print('data title:', data)
        elif self.lasttag == 'text' and self.textTag:
            self.text = data
            #print('data text:',data)

"""
Args:
    param1: Filename 

Returns:
    File content as text of the file passed as param 
"""
def getFileContent(fname):
    file_content = ''
    with open(fname) as f:
        content = f.readlines()
        for word in content:
            file_content += word.strip() + " "
        
    return file_content

"""
Args:
    param1: text 

Returns:
    tokens from WordPunctTokenizer
"""
def custom_tokenize(text):
    tokens = WordPunctTokenizer().tokenize(text)
    return tokens

"""
Returns:
    stop words from nltk corpus
"""
def getNltkCorpusStopWords():
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english')) 
    return stop_words

"""
Args:
    param1: word list as words 
    param2: stop_words

Returns:
    stemmed words after stemming, digits removal, short and stop words removal 
"""
def stemmingDigitsShortAndStopWordsRemoval(words, stop_words):
    
    stemmed_words = list()
    ps = PorterStemmer()
    for word in words:
        if word not in stop_words and word.isalnum() and not word.isdigit():
            stemmed_word = ps.stem(word)
            #stemmed_word = word
            if len(stemmed_word) > 2 and word not in stop_words:
                stemmed_words.append(stemmed_word)
        
    return stemmed_words


"""
Args:
    param1: filenames

Returns:
    A dictionary, key value pair of docs and its contents as text 
"""
def parse_SGML_text(filenames):
    
    all_files_content_dict = dict()
    html_parser = MyHTMLParser()
    
    for filename in filenames:
        file_content = getFileContent(filename)

        html_parser.feed(file_content)
        text = html_parser.text
        title = html_parser.title
        docno = html_parser.docno
        all_files_content_dict[docno.strip()] = (title + ' ' + text).lower()
        
    return all_files_content_dict


"""
Args:
    param1: all_files_content_dict: A dictionary, key value pair of 
    docs and its contents as text

Returns:
    A dictionary, key value pair of docs/queries and its contents as list of words
"""
def data_preprocessing(all_files_content_dict):
    
    stop_words = getNltkCorpusStopWords() 
    for key, value in all_files_content_dict.items():
        
        doc_id = key
        file_text = value
        
        tokens = custom_tokenize(file_text)
        words = stemmingDigitsShortAndStopWordsRemoval(tokens, stop_words)
        all_files_content_dict[doc_id] = words
    
    return all_files_content_dict


"""
Args:
    param1: queries_doc_path

Returns:
    A dictionary, key value pair of query id and its contents as text
"""
def get_queries_content_dict(queries_doc_path):
    
    queries_content_dict = dict()
    with open(queries_doc_path) as fp:  
        for cnt, line in enumerate(fp):
            queries_content_dict[str(cnt + 1)] = line
    fp.close()
    return queries_content_dict


"""
Args:
    param1: queries_content_dict: A dictionary, key value pair of 
    queries and its contents as list of words

Returns:
    A dictionary, key value pair of query id and its contents as text
"""
def calculate_queries_term_freq(queries_content_dict):
    
    queries_tf_dict = dict()
    
    for key, value in queries_content_dict.items():
        
        words = value
        query_file_vocab = dict()
        max_freq = 0
        for word in words:
            
            if word in query_file_vocab:
                query_file_vocab[word] += 1
            else:
                query_file_vocab[word] = 1
                
            if query_file_vocab[word] > max_freq:
                max_freq = query_file_vocab[word]
                
        queries_tf_dict[key] = [max_freq, query_file_vocab]
        
    return queries_tf_dict


"""
Args:
    param1: all_files_content_dict: A dictionary, key value pair of 
    docs and its contents as list of words

Returns:
    all_files_vocab, posting list
"""
def posting_list_creation(all_files_content_dict):
    
    doc_max_freq_dict = dict()
    all_files_vocab = dict()
    
    for key, value in all_files_content_dict.items():
        
        words = value
        file_vocab = {}
        max_word_freq = 0
        for word in words:

            if word in file_vocab:
                file_vocab[word] += 1

                doc_term_freq_dic = all_files_vocab[word][1]
                doc_term_freq_dic[key] += 1
                all_files_vocab[word] = [all_files_vocab[word][0], doc_term_freq_dic]
            else:
                file_vocab[word] = 1
                if word in all_files_vocab:
                    doc_term_freq_dict = all_files_vocab[word][1]
                    doc_term_freq_dict[key] = 1
                    all_files_vocab[word] = [(all_files_vocab[word][0]) + 1, doc_term_freq_dict]
                else:
                    all_files_vocab[word] = [1, {key: 1}]
                    
            if file_vocab[word] > max_word_freq:
                max_word_freq = file_vocab[word]
                    
        doc_max_freq_dict[key] = max_word_freq
        
    return all_files_vocab, doc_max_freq_dict


"""
Args:
    param1: all_files_vocab, posting list
    param2: doc_max_freq_dict
    param3: N, total number of dcos

Returns:
    documents length dictionay
"""
def calculate_docs_length(all_files_vocab, doc_max_freq_dict, N):

    docs_length_dict = dict()
    
    for key, value in all_files_vocab.items():
        
        df = all_files_vocab[key][0]
        
        for k, v in all_files_vocab[key][1].items():
            
            doc_id = k
            tf = v
            #max_freq = doc_max_freq_dict[doc_id]
            idf = math.log2(N/df)
            
            if doc_id in docs_length_dict:
                docs_length_dict[doc_id] += math.pow(tf * idf, 2)
            else:
                docs_length_dict[doc_id] = math.pow(tf * idf, 2)
                
    return docs_length_dict


def get_value_based_sorted_dict(val_dict):
    from collections import OrderedDict
    sorted_dict = OrderedDict(sorted(val_dict.items(), key=lambda x: x[1],  reverse=True))
    return sorted_dict


"""
Args:
    param1: relevance doc path
    param3: N, total number of dcos

Returns:
    A dictionary of relevant docs for each query
"""
def getRelevantQuerydocs(relevance_doc_path):
    
    queries_relevant_docs_dict = dict()
    with open(relevance_doc_path) as fp:  
        for line in fp:
            line_text_list = line.strip().split()
            q_id = line_text_list[0]
            rel_doc_no = line_text_list[1]
            if q_id in queries_relevant_docs_dict:
                queries_relevant_docs_dict[q_id].add(rel_doc_no)
            else:
                queries_relevant_docs_dict[q_id] = {rel_doc_no}
                
    return queries_relevant_docs_dict


"""
Args:
    param1: docs_length_dict
    param2: all_files_vocab
    param3: doc_max_freq_dict
    param4: queries_tf_dict
    param5: N, number of all docs

Returns:
    A dictionary, key value pair of queries and docs(sorted based on cosine similarity)
"""
def calculate_cosine_similarities(docs_length_dict, all_files_vocab, doc_max_freq_dict, queries_tf_dict, N):
    
    all_queries_cosine_similarities = dict()
    
    for key in queries_tf_dict:
        q_id = key
        q_cosine_scores_dict = calculate_query_cosine_similarities(q_id, docs_length_dict, all_files_vocab, doc_max_freq_dict, queries_tf_dict, N)
        q_Sorted_cosine_scores_dict = get_value_based_sorted_dict(q_cosine_scores_dict)
        
        all_queries_cosine_similarities[q_id] = [key for key in q_Sorted_cosine_scores_dict]
        
    return all_queries_cosine_similarities

"""
Args:
    param1: q_id
    param2: docs_length_dict
    param3: all_files_vocab
    param4: doc_max_freq_dict
    param5: queries_tf_dict
    param6: N, number of all docs

Returns:
    A dictionary of cosine_similarities of docs for a query
"""
def calculate_query_cosine_similarities(q_id, docs_length_dict, all_files_vocab, doc_max_freq_dict, queries_tf_dict, N):
    
    #q_max_freq = queries_tf_dict[q_id][0]
    q_tf_dict = queries_tf_dict[q_id][1]

    q_cosine_scores_dict = dict()
    q_term_weight_sq_sum = 0
    
    for k, v in q_tf_dict.items():

        word = k
        q_term_freq = v
        
        if word in all_files_vocab:
        
            df = all_files_vocab[word][0]
            idf = math.log2(N/df)
            
            q_term_weight = q_term_freq * idf
            q_term_weight_sq_sum += math.pow(q_term_weight, 2)

            for key, value in all_files_vocab[word][1].items():
                
                doc_id = key
                doc_tf = value

                #doc_max_freq = doc_max_freq_dict[doc_id]
                if doc_id in q_cosine_scores_dict:
                    q_cosine_scores_dict[doc_id] += q_term_weight * (doc_tf * idf)
                else:
                    q_cosine_scores_dict[doc_id] = q_term_weight * (doc_tf * idf)
                
    for ke, val in q_cosine_scores_dict.items():
        q_cosine_scores_dict[ke] = val / math.sqrt(docs_length_dict[ke] * q_term_weight_sq_sum)
                
    return q_cosine_scores_dict

"""
Calculate and prints average precision, recall of all queries for topN ranked documents
Args:
    param1: all_queries_cosine_similarities
    param2: queries_relevant_docs_dict
    param3: topN
"""
def calculate_precision_recall(all_queries_cosine_similarities, queries_relevant_docs_dict, topN):
    
    q_N = len(all_queries_cosine_similarities)
    avg_precision = 0
    avg_recall = 0
    precision_recall_dict = dict()
    for key in all_queries_cosine_similarities:
        
        q_id = key
        retrieved_q_docs = all_queries_cosine_similarities[q_id]
        retrieved_q_docs = retrieved_q_docs[:topN]
        
        relevant_q_docs = queries_relevant_docs_dict[q_id]
        TP = 0
        for doc in retrieved_q_docs:
            if doc in relevant_q_docs:
                TP += 1
        recall = TP/len(relevant_q_docs)
        precision = TP/topN
        precision_recall_dict[q_id] = [precision, recall]

        avg_precision += precision
        avg_recall += recall
        
    avg_precision = avg_precision/q_N
    avg_recall = avg_recall /q_N
    
    print(' ')
    print("Precision Recall for top ",topN," retrieved documents in rank list: ")
    for k, val in precision_recall_dict.items():
        print("Query: ",k,"     Pr: ",val[0],"      Re: ",val[1])
    
    print('Average Precision for ',topN,' docs: ',avg_precision)
    print('Average Recall for ',topN,' docs: ',avg_recall)

    return


"""
Retrieves relevant docs from a collection for given queries and
calculates average precision and recall based on given relevance text
Args:
    param1: data_path, path of data collection
    param2: queries_doc_path, path of queries text file
    param3: path of relevance text file
"""
def relevant_docs_retrieval(data_path, queries_doc_path, relevance_doc_path):
    
    filenames = glob.glob(data_path)
    N = len(filenames)
    
    all_files_content_dict = parse_SGML_text(filenames)
    queries_content_dict = get_queries_content_dict(queries_doc_path)
    
    all_files_content_dict = data_preprocessing(all_files_content_dict)
    queries_content_dict = data_preprocessing(queries_content_dict)
    
    all_files_vocab, doc_max_freq_dict = posting_list_creation(all_files_content_dict)
    docs_length_dict = calculate_docs_length(all_files_vocab, doc_max_freq_dict, N)
 
    queries_tf_dict = calculate_queries_term_freq(queries_content_dict)
    
    all_queries_cosine_similarities = calculate_cosine_similarities(docs_length_dict, all_files_vocab, doc_max_freq_dict, queries_tf_dict, N)
    queries_relevant_docs_dict = getRelevantQuerydocs(relevance_doc_path)

    calculate_precision_recall(all_queries_cosine_similarities, queries_relevant_docs_dict, 10)
    calculate_precision_recall(all_queries_cosine_similarities, queries_relevant_docs_dict, 50)
    calculate_precision_recall(all_queries_cosine_similarities, queries_relevant_docs_dict, 100)
    calculate_precision_recall(all_queries_cosine_similarities, queries_relevant_docs_dict, 500)

    return


if __name__ == "__main__":
    
# =============================================================================
#     data_path = 'cranfield.tar/cranfieldDocs/*'
#     queries_doc_path = 'queries.txt'
#     relevance_doc_path = 'relevance.txt'
# =============================================================================
    
    data_path = sys.argv[1]
    queries_doc_path = sys.argv[2]
    relevance_doc_path = sys.argv[3]
    relevant_docs_retrieval(data_path, queries_doc_path, relevance_doc_path)