import nltk
nltk.help.upenn_tagset('MD')

# POS tagging with NLTK
text11 = "TORUS (Targeted Observation by Radars and UAS of Supercells) is a nomadic field campaign during the spring storm seasons (May and June) of 2019 and 2010 over a domain covering much of the central United Stats where there exists significant point probabilities of tornado-bearing supercell storms. "
text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13)