#this is a keyword extraction algorithm
#this program requires python3 or higher to run
#The following Dependancies are requierd:
# retinasdk, NumPy, SciPy, gensim, nltk
#using the command line do "pip3 install nltk" or "sudo pip3 install nltk" for each package 
#to run place the text file you the keywords from in a directory with this .py file
#to call it from the command line do: "python3 keywords_extraction.py" and follow the instructions 

################################################################################
import retinasdk
import time
import os
import spacy
import re
import fileinput
from gensim.summarization import keywords
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from string import punctuation


#################################################################################

def checkpoint(check):
  print("checkpoint " + check)
  print("\n")
  time.sleep(3)

def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        print("{} does not exist ".format(filename))
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()

    with open(filename, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines) 


#string to list converter
def Convert(string): 
    li = list(string.split("-")) 
    return li 
#retinasdk client connection
liteClient = retinasdk.LiteClient("1d5b1cc0-aa65-11e9-8f72-af685da1b20e")

#################################################################################

#make sure directory is cleaned from last run
#if os.path.isfile("keywords.txt"):
#	os.remove("keywords.txt")

if os.path.isfile("clean.txt"):
	os.remove("clean.txt")
if os.path.isfile("keywords.txt"):
	os.remove("keywords.txt")
if os.path.isfile("temp.txt"):
	os.remove("temp.txt")
if os.path.isfile("outfile.txt"):
  os.remove("outfile.txt")
if os.path.isfile("temp2.txt"):
  os.remove("temp2.txt")
if os.path.isfile("keywords.txt"):
  os.remove("keywords.txt")


outfile = open("keywords.txt", "w")



#welocme message and basic information
print ("Welcome to Keyword Extractor 6.0")
print ("Please have the text you need keywords extracted from ready in a text file in the same directory")
print (" ")
#take in file name
file_name = raw_input("Enter file name: ")
print (" ")
#open file and read in
print ("Opening file: ", file_name)
abstract_file = open(file_name, "r+") #opens file as r/w only
# print(abstract_file.readable())
print("\n")

with open(abstract_file) as f:
  f = f.read().replace(' ', '').replace('\n','').lower()
  f = f.strip(string.punctuation)
  f.write(f)

abstract = abstract_file.read()																										


#################################################################################

#extract keywords
print (abstract)
print (" ")
print (" ")

print ("Extracting Keywords")

list_1 = liteClient.getKeywords(abstract) #list 1 is a list
lite_keywords = ' '.join(list_1) #list 1 becomes a string
text_en = text_en = (abstract)
extracted_keywords = keywords(text_en,words = 20,scores = False, lemmatize = True) #spits out list
lite_keywords_lines = lite_keywords.splitlines()

#NER Implmementation

abstract_NER = abstract

spacy_nlp = spacy.load('en')
document = spacy_nlp(abstract_NER)

temp_file = "temp.txt"
file_temp = open(temp_file, "a")
for name in lite_keywords_lines:
  file_temp.write(name)
  file_temp.write(" ")
  # file_temp.write("\n")


file_temp.write(" \n".join(lite_keywords_lines)) #spits out string
file_temp.write(" ") #needed to seaperate keyword group 1 and 2
file_temp.write("\n")
file_temp.write(extracted_keywords)	#spits out lists
file_temp.write(" ")
file_temp.write("\n")


for element in document.ents:
  temp_write = str(element)
  file_temp.write(temp_write)
  file_temp.write("\n")

abstract_file.close()
file_temp.close()


#################################################################################

#clean up fomating and put all into a list

final_file_name = "clean.txt"

final_file = open(final_file_name, "w")

with open(temp_file,'r') as f:
    for line in f:
        for word in line.split():
           final_file.write(word)
           final_file.write(" \n")

final_file.close()

#################################################################################
#remove duplicates from all keywords

content = open('clean.txt', 'r') 
content_set = set(content)

lines_seen = set() # holds lines already seen
for line in open("clean.txt", "r"):
    if line.lower() not in lines_seen: # not a duplicate
        outfile.write(line)
        lines_seen.add(line)
outfile.close()

cleandata = open('temp2.txt', 'w')

for line in content_set:  #moves everything to temp file for cleaning
  cleandata.write(line.lower())
cleandata.close()   

#################################################################################

#deleted temp files 
if os.path.isfile("clean.txt"):
  os.remove("clean.txt")
if os.path.isfile("keywords.txt"):
  os.remove("keywords.txt")
if os.path.isfile("temp.txt"):
  os.remove("temp.txt")


#################################################################################

#removes duplicate words

infile = open("temp2.txt","r")
wordsDict = {}
for line in infile:
    addBoolean = True
    for word in wordsDict:
        if word == line:
            addBoolean = False
            break
    if addBoolean:
        wordsDict[line] = None
infile.close() 

outfile_name = "outfile.txt"   
outfile = open(outfile_name,"w")
for word in wordsDict:
    outfile.write(word+'\n')

#################################################################################
#removes numbers

table = str.maketrans(dict.fromkeys('0123456789'))

with open("temp2.txt", 'r') as f_in:
    data = f_in.read()
data = data.translate(table)
with open(outfile_name, 'w') as f_out:
    f_out.write(data)

for line in fileinput.FileInput(outfile_name,inplace=1):
    if line.rstrip():
        print (line)
        
if os.path.isfile("temp2.txt"):
  os.remove("temp2.txt")

#################################################################################

#removes duplicate words

infile = open("outfile.txt","r")
wordsDict = {}
for line in infile:
    addBoolean = True
    for word in wordsDict:
        if word == line:
            addBoolean = False
            break
    if addBoolean:
        wordsDict[line] = None
infile.close()    
outfile = open("keywords.txt","w")
for word in wordsDict:
    outfile.write(word+'\n')

if os.path.isfile("outfile.txt"):
  os.remove("outfile.txt")
outfile.close()
#################################################################################

#remove empty white space

remove_empty_lines("keywords.txt")
 