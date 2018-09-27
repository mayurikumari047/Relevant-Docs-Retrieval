# Relevant-Docs-Retrieval
Retrieving relevant docs from a corpus for given queries

Functions implemented for this program: 
1.	Get the file names list: Used glob.glob() function to get the list of all filenames from a given path.
2.	Get all files content: Reads each file and returns the content of each files.
3.	Parse SGML text: Parses files using HTMLParser to get only title and text content of each file and store it in a dictionary as value for corresponding Doc Id as key.
4.	Get the queries content:  Reads each line as a query from given queries.txt file and stores all the queries in a dictionary as query id and query content.
5.	Data Preprocessing: Performs data preprocessing on each file content and each query content.
a.	Tokenization on space, Punctuation removal and convert to lower case: Used Nltk WordPunctTokenizer() api to tokenize the text on space, isalnum() api to remove punctuations from the tokens and lower() function to make all words to lower case.
b.	Stop words removal: Gets a list of words and stop words, removes stop words from given list of words by comparing with list of stop words.
c.	Stemming: Takes list of words as arguments, uses NLTK Porter stemmer on these words and returns a list of stemmed words.
d.	Stop words removal after stemming: Removes stop words from stemmed words.
e.	Remove digits: Removes digits from all the words.
f.	Remove short words: Remove any word of length less than 3 characters.
6.	Posting list creation: Prepares posting list for all unique words in the collection. Reads each file word list and creates a dictionary of all unique words in the collection as key and value would be a list of document frequency of that word and a dictionary of term frequency of that word for docs in which that word appeared at least once.
7.	Calculate docs length: Scans the prepared posting list to get the document frequency and term frequency of each word for each document and calculates the document length by summing squared term weight (tf * idf) of each term in the doc.
8.	Calculate queries term frequency: Calculates the term frequencies of each query term for all the queries. Prepares a dictionary of query id and query vocabulary which itself is a dictionary of query term and its term frequency in query.
9.	Calculate cosine similarities: Scans queries term frequencies dictionary, gets the query id and calls calculate_query_cosine_similarities() api to get the q_cosine_scores_dict. Sorts this dictionary based on value which is here cosine scores of each doc for a query. Prepares a dictionary of query id as key and list of doc ids which are sorted based on cosine scores as value and returns this dictionary.
10.	Calculate query cosine similarity: Gets the query term frequency dictionary for given query. For each term in query term frequency dictionary, calculates the term weight by taking df from posting list and tf from q_tf_dict. For each pair doc and query term in posting list, compute score for the doc by doing Scores[d] += wft,d × wt,q. At the end, divide the doc cosine scores by its doc length for each doc. 
11.	Calculate precision and recall: Gets the queries cosine similarities dictionary and queries relevant docs dictionary and topN number. For each query, compares if doc id in the retrieved topN doc list is present in the actual relevant doc list for that query from given relevance text file and calculates the True positive docs. Calculates precision by dividing this true positive number by topN and recall by dividing this true positive number by total actual relevant document for that query. Calculate the average precision and recall for all the queries for given topN retrieved doc number and prints it.

Time complexity for this whole program is O(m*N) which is the time complexity of reading the entire collection.
