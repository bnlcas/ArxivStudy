# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 18:26:08 2016

@author: BenLucas
# WebScrapes Arxiv for abstracts of various articles by field
# Generates a list of Abstracts and a list of primary topics
# for more data, change the max_results variable.
"""

import urllib
import csv
import feedparser
from time import sleep


# Base api query url
base_url = 'http://export.arxiv.org/api/query?'




# Search parameters

topics = ['Astrophysics', 'Condensed Matter', 'General Relativity and Quantum Cosmology', 
              'High Energy Physics - Experiment', 'High Energy Physics - Lattice',
              'High Energy Physics - Phenomenology','High Energy Physics - Theory',
              'Mathematical Physics','Nonlinear Sciences', 'Nuclear Experiment','Nuclear Theory',
              'Physics','Quantum Physics','Mathematics','Computer Science','Quantitative Biology',
              'Quantitative Biology', 'Statistics']
              
topics = ['math', 'stat', 'cs', 'astro-ph', 'cond-mat', 'gr-qc', 
          'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math-ph', 'nlin' 
          'nucl-ex', 'nucl-th' ,'physics', 'quant-ph', 'q-bio', 'q-fin']
          
topics = []
file_name = '/Volumes/NO NAME/Arxiv Study/ArxivAbrevs.csv'
with open(file_name,'rb') as d:
    reader = csv.reader(d)
    for row in reader:
        topics.append(row[0])


abstracts = []
primary_topics = []
for topic in topics:
    search_query = 'all:' + topic # search for electron in all fields
    start = 0                     # retreive the first 5 results
    max_results = 1000
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, start,max_results)
    # Opensearch metadata such as totalResults, startIndex, 
    # and itemsPerPage live in the opensearch namespase.
    # Some entry metadata lives in the arXiv namespace.
    # This is a hack to expose both of these namespaces in
    # feedparser v4.1
    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'
    # perform a GET request using the base_url and query
    response = urllib.urlopen(base_url+query).read()
    
    # parse the response using feedparser
    feed = feedparser.parse(response)
    
    # Run through each entry, and print out information
    for entry in feed.entries:
        tag = entry.tags[0]['term']
        #print tag
        if topic in tag:
            primary_topics.append(tag)
            abstracts.append(entry.summary)
            
    
root_dir = '/Volumes/NO NAME/Arxiv Study/'

utf8_inds = []
for i, a in enumerate(abstracts):
    try:
        a.decode('utf-8')
        utf8_inds.append(i)
    except UnicodeError:
        aaaa = 1
        
abstracts_test = [a for i, a in enumerate(abstracts) if i in utf8_inds]
topics_test = [a for i, a in enumerate(primary_topics) if i in utf8_inds]
        
    
testwith open(root_dir + 'Abstract_List.csv', 'wb') as f:
    writer = csv.writer(f,)
    writer.writerows(abstracts_test)

with open(root_dir + 'Topic_List.csv','wb') as f:
    writer = csv.writer(f,)
    writer.writerows(topics_test)
    
abstracts = loadcsv2list(root_dir+'Abstract_List.csv')
abstracts = [''.join(a) for a in abstracts]
abstracts = [str.replace(a,'\n',' ') for a in abstracts]

Primary_Topics = loadcsv2list(root_dir+'Topic_List.csv')
Primary_Topics = [''.join(a) for a in Primary_Topics]
