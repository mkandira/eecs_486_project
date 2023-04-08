import sys, re, os, math
from preprocess import removeSGML, tokenizeText, removeStopwords, stemWords
from porter import PorterStemmer
import glob
import string
from emoji import UNICODE_EMOJI
import csv

def is_emoji(s):
    return s in UNICODE_EMOJI

def trainNaiveBayes(training_file):
    # Initialzing vocab_set
    vocab_set = []
    
    predicted_class_information_value = {}
    predicted_class_information_value["num_docs"] = 0
    predicted_class_information_value["total_words"] = 0
    predicted_class_information_value["tf"] = {}
   
    predicted_class_information = {}
    predicted_class_information["gen_x"] = predicted_class_information_value
    predicted_class_information["millenial"] = predicted_class_information_value
    predicted_class_information["gen_z"] = predicted_class_information_value
    predicted_class_information["boomer"] = predicted_class_information_value
    
    with open(training_file, 'r') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            print(row)
            # Opening the file
            tweet_content = row[1]
            
            # Removing the SGML tags
            tweet_content = removeSGML(tweet_content)

            # Tokenizing text
            tweet_content = tokenizeText(tweet_content)

            #Getting filename
            tweet_class = row[2]
                
            # If file is in class "true"
            if tweet_class == "gen_z":
                # Incrementing counts of gen z documents
                predicted_class_information["gen_z"]["num_docs"] += 1

                # Looping through file content
                for word in tweet_content:
                    # Incrementing number of words
                    predicted_class_information["gen_z"]["total_words"] += 1
                    word = word.lower()

                    # Check if word is emoji
                    if is_emoji(word):
                        word = "general_emoji_encoding"
                    
                    # Adding to total docs
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)
                    
                    if word in predicted_class_information["gen_z"]["tf"] and word not in string.punctuation:
                        predicted_class_information["gen_z"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["gen_z"]["tf"][word] = 1
            # If file is millenial
            elif tweet_class == "millenial":
                # Incrementing counts of millenial documents
                predicted_class_information["millenial"]["num_docs"] += 1

                # Looping through file content
                for word in tweet_content:
                    # Incrementing number of words
                    predicted_class_information["millenial"]["total_words"] += 1
                    word = word.lower()
                    # Check if word is emoji
                    if is_emoji(word):
                        word = "general_emoji_encoding"
                    
                    # Adding to total docs
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)
                    
                    if word in predicted_class_information["millenial"]["tf"] and word not in string.punctuation:
                        predicted_class_information["millenial"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["millenial"]["tf"][word] = 1
                        
            elif tweet_class == "gen_x":
                # Incrementing counts of millenial documents
                predicted_class_information["gen_x"]["num_docs"] += 1

                # Looping through file content
                for word in tweet_content:
                    # Incrementing counts of gen x words
                    predicted_class_information["gen_x"]["total_words"] += 1
                    word = word.lower()

                    # Check if word is emoji
                    if is_emoji(word):
                        word = "general_emoji_encoding"
                    
                    # Adding to total docs
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)
                    
                    if word in predicted_class_information["gen_x"]["tf"] and word not in string.punctuation:
                        predicted_class_information["gen_x"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["gen_x"]["tf"][word] = 1
                
            elif tweet_class == "boomer":
                # Incrementing counts of millenial documents
                predicted_class_information["boomer"]["num_docs"] += 1

                # Looping through file content
                for word in tweet_content:
                    predicted_class_information["boomer"]["total_words"] += 1
                    word = word.lower()

                    # Check if word is emoji
                    if is_emoji(word):
                        word = "general_emoji_encoding"
                    
                    # Adding to total docs
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)

                    
                    if word in predicted_class_information["boomer"]["tf"] and word not in string.punctuation:
                        predicted_class_information["boomer"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["boomer"]["tf"][word] = 1
    
    return predicted_class_information, len(vocab_set)

            

def testNaiveBayes(test_file_path, predicted_class_information, vocab_size):
    # Opening the file
    file_content = open(test_file_path, "r").read()
    
    # Removing the SGML tags
    file_content = removeSGML(file_content)

    # Tokenizing text
    file_content = tokenizeText(file_content)
    # TODO: Check for emoji

    

if __name__ == "__main__":
    # Getting the filename from the command line
    test_file = os.fsdecode(sys.argv[1])
    training_file = os.fsdecode(sys.argv[2])

    # Initializing list of files
    test_filename_content = {}
    training_filename_paths = []


    # Looping through filename paths
    # Training a Naive Bayes classifier:
    # For each class we need: number of documents in that class, number of words in that class, word/count dictionary for each class
    # Size of vocab
    # predicted_class_information => key = class_name, value = {"num_docs": 5, "total_words": 1000, "tf": {}}
    predicted_class_information, vocab_size = trainNaiveBayes(training_file)
  
   
        
    # Testing the Naive Bayes classifier:
    for test_file_path in test_filename_paths:
        predicted_class = testNaiveBayes(test_file_path, predicted_class_information, vocab_size)
        output_file.write(str(test_file_path) + " " + str(predicted_class) + '\n')
    
        #Calculating accruancy
        if predicted_class == test_file_path.split("/")[-1][:4]:
            accuracy += 1
        
    print(accuracy/len(filename_paths))

    # # Assuming class is true
    # p_true = num_true_docs/num_true_docs+num_fake_docs

    # # Looping through words in content
    # for word in file_content:
    #     n_k = 0
    #     if word in true_vocab_set:
    #         n_k = true_vocab_set[word]
    #     p_true *= ((n_k + 1)/(total_words_true+len(vocab_set)))

    # # Assuming class is false
    # p_false = num_true_docs/num_true_docs+num_fake_docs

    # # Looping through words in content
    # for word in file_content:
    #     n_k = 0
    #     if word in fake_vocab_set:
    #         n_k = fake_vocab_set[word]
    #     p_false *= ((n_k + 1)/(total_words_fake+len(vocab_set)))
    
    # # If the probability of word being class true is greater than false, return true
    # if p_true > p_false:
    #     return "true"
    # # Else return false
    # return "fake"








        


    
