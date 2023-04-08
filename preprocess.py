
import re
import sys
import os
from porter import PorterStemmer
import string

def removeSGML(s):
    new_s = re.sub(re.compile('<.*?>'),'',s)
    return new_s

def tokenizeText(s):
    tokens = s.split()
    punctuation = [" ", ",",";", "!", "?", "'", "."]
    result = []
    for token in tokens:
        # Case 1: token is not single character
        if len(token) > 1:
            # Case 1.a: token has "." at the end and it is not an abbreviation or acronym
            if token[-1] == "." and token.count(".") == 1:
                result.append(token[0:len(token)-1])
                result.append(token[-1])

            # Case 1.b: remove end punction
            elif token[-1] in punctuation:
                result.append(token[0:len(token)-1])
                result.append(token[-1])   
            
            # Case 1.c: handle 1 letter contraction and posessive
            elif len(token) >=3 and token[-2] == "'":
                # Ex: he's (contraction)
                if token[-1] == "s" and not token[0].isupper():
                    result.append(token[0:len(token)-2])
                    result.append("is")
                # Ex: Martha's (possession)
                elif token[-1] == "s" and not token[0].isupper():
                    result.append(token[0:len(token)-2])
                    result.append(token[-2:])
                # Ex: I'm
                elif token[-1] == "m":
                    result.append(token[0:len(token)-2])
                    result.append("am")
                # Ex" She'd
                elif token[-1] == "d":
                    result.append(token[0:len(token)-2])
                    result.append("would")
                # Ex: Can't
                elif token[-1] == "t" and token[-3] == "n":
                    result.append(token[0:len(token)-2])
                    result.append("not")

            # Case 1.d: handle 2 letter contraction
            elif len(token) >=4 and token[-3] == "'":
                if token[-2:] == "re":
                    result.append(token[0:len(token)-3])
                    result.append("are")
                elif token[-2:] == "ll":
                    result.append(token[0:len(token)-3])
                    result.append("will")
                elif token[-2:] == "ve":
                    result.append(token[0:len(token)-3])
                    result.append("have")
            
            # Case 1.e: word remains unchanged
            else:
                result.append(token)
 
        # Case 2: token is a single character
        else:
            # Case 2.a: token is not a punctuation
            if token[0] not in punctuation:
                result.append(token) 
    return result

def removeStopwords(tokens):
    stopWords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'few', 'from', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'to', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your']
    result = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in stopWords and token_lower not in string.punctuation:
            result.append(token)
    return result

# d. Function that stems the words.
def stemWords(tokens):
    stemmed_words = []
    Port = PorterStemmer()

    for word in tokens:
       stemmed_words.append(Port.stem(word, 0, len(word)-1))
    
    return stemmed_words

if __name__ == "__main__":
    stemmer = PorterStemmer()
    # Grabbing the filename from the command line
    folder = os.fsdecode(sys.argv[1])

    total_words = 0
    vocabulary = {}

    # Looping through the files in the director
    for file in os.listdir(folder):
        # Opening the file
        filename = os.fsdecode(file)
        path = str(folder) + "/" + str(filename)
        file_content = open(path, "r").read()
        
        # Removing the SGML tags
        file_content = removeSGML(file_content)

        # Tokenizing text
        file_content = tokenizeText(file_content)

        # Removing stop words
        file_content = removeStopwords(file_content)

        # Stemming words
        file_content = stemWords(file_content)

        # Stemming words
        # stemmed_file_content = [stemmer.stem(word) for word in file_content]

        # Incrementing total word size
        total_words += len(file_content)

        # Incrementing total unique words
        for word in file_content:
            if word not in vocabulary:
                vocabulary[word] = 1
            else:
                vocabulary[word] += 1
    
    # Sorting vocabulary by value 
    sorted_vocabulary_frequency = sorted(vocabulary.items(), key=lambda x:x[1])
    sorted_vocabulary_frequency.reverse()

    #Outputting information to file
    output_file = open('preprocess.output', 'w')
    print('Words: ' + str(total_words), file=output_file)
    print('Vocabulary: ' + str(len(vocabulary)), file=output_file)
    
    counter = 0
    for word in sorted_vocabulary_frequency:
        if word[0] not in string.punctuation:
            print(str(word[0]) + ' ' + str(word[1]), file=output_file)
            counter += 1
        if counter > 50:
            break
    
           
    # total = 0
    # countWords = 0
    # while total < (145658/4):
    #     total += sorted_vocabulary_frequency[countWords][1]
    #     countWords += 1
    # print(countWords, total)


    
    


    




    
