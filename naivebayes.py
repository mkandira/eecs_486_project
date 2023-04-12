import sys, os
from preprocess import tokenizeText, removeStopwords, stemWords
from porter import PorterStemmer
import string
import emoji
import csv

def trainNaiveBayes(training_file):
    # Initialzing vocab_set
    vocab_set = []

    # Initializing a class for the information needed to predict each generation class
    predicted_class_information_value = {}
    predicted_class_information_value["num_docs"] = 0
    predicted_class_information_value["total_words"] = 0
    predicted_class_information_value["tf"] = {}
   
    # Initialzing a dictionary where key is generation and value is information needed to predict that generation
    predicted_class_information = {}
    predicted_class_information["gen_x"] = dict(predicted_class_information_value)
    predicted_class_information["millenial"] = dict(predicted_class_information_value)
    predicted_class_information["gen_z"] = dict(predicted_class_information_value)
    predicted_class_information["boomer"] = dict(predicted_class_information_value)
    
    # Opening training file
    with open(training_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        
        index = 0
        for row in reader:
            index += 1
            # Getting tweet content
            tweet_content = row[1]
            
            # Tokenizing tweet
            tweet_content = tokenizeText(tweet_content)

            # Getting tweet classifcation
            tweet_class = row[2]

            # If tweet is in classified as "gen z"
            if "gen z" in tweet_class:
                # Incrementing the counts the number of documents/tweets with the gen z classification
                predicted_class_information["gen_z"]["num_docs"] += 1

                # Looping through tweet content
                for word in tweet_content:
                    # Incrementing number of words in the gen z classification
                    predicted_class_information["gen_z"]["total_words"] += 1
                    
                    # Converting word to lower case
                    word = word.lower()

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
                    # Adding word to vocab set if not present 
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)
                    
                    # Adding word to tf collection of the gen z class
                    if word in predicted_class_information["gen_z"]["tf"] and word not in string.punctuation:
                        predicted_class_information["gen_z"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["gen_z"]["tf"][word] = 1
            # If tweet is in classified as "millenial"
            elif "millenial" in tweet_class:
                # Incrementing the counts the number of documents/tweets with the millenial classification
                predicted_class_information["millenial"]["num_docs"] += 1

                # Looping through tweet content
                for word in tweet_content:
                    # Incrementing number of words in the millenial classification
                    predicted_class_information["millenial"]["total_words"] += 1

                    # Converting word to lower case
                    word = word.lower()

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
                    # Adding word to vocab set if not present 
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)
                    
                    # Adding word to tf collection of the millenial class
                    if word in predicted_class_information["millenial"]["tf"] and word not in string.punctuation:
                        predicted_class_information["millenial"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["millenial"]["tf"][word] = 1
            # If tweet is in classified as "gen x"           
            elif "gen x" in tweet_class:
                # Incrementing the counts the number of documents/tweets with the gen x classification
                predicted_class_information["gen_x"]["num_docs"] += 1

                # Looping through tweet content
                for word in tweet_content:
                    # Incrementing number of words in the gen x classification
                    predicted_class_information["gen_x"]["total_words"] += 1
                    
                    # Converting word to lower case
                    word = word.lower()

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
                    # Adding word to vocab set if not present 
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)
                    
                    # Adding word to tf collection of the gen x class
                    if word in predicted_class_information["gen_x"]["tf"] and word not in string.punctuation:
                        predicted_class_information["gen_x"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["gen_x"]["tf"][word] = 1
            # If tweet is in classified as "boomer"   
            elif "boomer" in tweet_class:
                # Incrementing the counts the number of documents/tweets with the boomer classification
                predicted_class_information["boomer"]["num_docs"] += 1

                # Looping through tweet content
                for word in tweet_content:
                    # Incrementing number of words in the boomer classification
                    predicted_class_information["boomer"]["total_words"] += 1

                    # Converting word to lower case
                    word = word.lower()

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
                    # Adding word to vocab set if not present 
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)

                    # Adding word to tf collection of the boomer class
                    if word in predicted_class_information["boomer"]["tf"] and word not in string.punctuation:
                        predicted_class_information["boomer"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["boomer"]["tf"][word] = 1
            else:
                print(tweet_class)
                print(index)
    print(index)
    return predicted_class_information, len(vocab_set)

            
# input: the file path to be used for test;
# input: the output produced by trainNaiveBayes;
# output: predicted class (the string “true” or the string “fake”. You can assume these to be the only classes to be predicted
def testNaiveBayes(input_tweet: str, predicted_class: dict, vocab_size: int) -> str:
    """Take in class probabilites and total vocab size and test on input tweet."""
    
    # Initialzing each classifications probability to be equal to the proportion of tweets what that classification
    total_docs = predicted_class["gen_z"]["num_docs"] + predicted_class["millenial"]["num_docs"] + predicted_class["gen_x"]["num_docs"] + predicted_class["boomer"]["num_docs"]
    gen_z_probability = predicted_class["gen_z"]["num_docs"]/total_docs
    gen_x_probability =  predicted_class["gen_x"]["num_docs"]/total_docs
    millenial_probability =  predicted_class["millenial"]["num_docs"]/total_docs
    boomer_probability =  predicted_class["boomer"]["num_docs"]/total_docs

    
    # Looping through each token in the input tweet
    for token in input_tweet:
        # If token is emoji, convert to plain text
        token = emoji.demojize(token)

        # Adding to gen z probability
        # If the token is not present in the gen z tf dictionary, we multiply by 1/(total gen z words + vocab size)
        if token not in predicted_class["gen_z"]["tf"]:
            prob = 1 / (predicted_class["gen_z"]["total_words"] + vocab_size)
            gen_z_probability *= float(prob)
        # Else, we multiply by (1 + token tf)/(total gen z words + vocab size)
        else:
            prob = (predicted_class["gen_z"]["tf"][token] + 1) / \
            float(predicted_class["gen_z"]["total_words"] + vocab_size)
            gen_z_probability *= float(prob)

        # Adding to millenial probability
        # If the token is not present in the millenial tf dictionary, we multiply by 1/(total millenial words + vocab size)
        if token not in predicted_class["millenial"]["tf"]:
            prob = 1 / float(predicted_class["millenial"]["total_words"] + vocab_size)
            millenial_probability *= float(prob)
        # Else, we multiply by (1 + token tf)/(total millenial words + vocab size)
        else:
            prob = (predicted_class["millenial"]["tf"][token] + 1) / \
            float(predicted_class["millenial"]["total_words"] + vocab_size)
            millenial_probability *= float(prob)

        # Adding to  gen x probability
        # If the token is not present in the gen x tf dictionary, we multiply by 1/(total gen x words + vocab size)
        if token not in predicted_class["gen_x"]["tf"]:
            prob = 1 / (predicted_class["gen_x"]["total_words"] + vocab_size)
            gen_x_probability *= float(prob)
        # Else, we multiply by (1 + token tf)/(total gen x words + vocab size)
        else:
            prob = (predicted_class["gen_x"]["tf"][token] + 1) / \
            float(predicted_class["gen_x"]["total_words"] + vocab_size)
            gen_x_probability *= float(prob)

        # Adding to boomer probability
        # If the token is not present in the boomer tf dictionary, we multiply by 1/(total boomer words + vocab size)
        if token not in predicted_class["boomer"]["tf"]:
            prob = 1 / (predicted_class["boomer"]["total_words"] + vocab_size)
            boomer_probability *= float(prob)
        # Else, we multiply by (1 + token tf)/(total boomer words + vocab size)
        else:
            prob = (predicted_class["boomer"]["tf"][token] + 1) / \
            float(predicted_class["boomer"]["total_words"] + vocab_size)
            boomer_probability *= float(prob)


    # Finding our highest probability and returning the class which has the highest probability
    maxProb = max(gen_z_probability,gen_x_probability,millenial_probability,boomer_probability)
    if gen_z_probability == maxProb: 
        return 'gen z'
    elif gen_x_probability == maxProb:  
        return 'gen x' 
    elif millenial_probability == maxProb:
        return 'millenial'
    else:
        return 'boomer'
    

if __name__ == "__main__":
    # Getting the testing and training filename from the command line
    training_file = os.fsdecode(sys.argv[1])
    test_file = os.fsdecode(sys.argv[2])
    

    # Calling the training function Functins
    # Returns: predicted_class_information => key = class_name, value = {"num_docs": 5, "total_words": 1000, "tf": {}}
    # Returns: vocab i=size
    predicted_class_information, vocab_size = trainNaiveBayes(training_file)
  
        
    # Testing the Naive Bayes classifier:

    # Initialzing variable to keep track of the number of tweet predictions we got right 
    accuracy = 0

    # Initialzing variable to keep track of the total number of tweets
    total_tweets = 0

    # Opening test file
    with open(test_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        
        # Looping through each tweet
        for row in reader:
            # Incrementing tweet count
            total_tweets += 1
            
            # Getting the tweet content
            tweet_content = row[1]

            # Tokenizing tweet
            tweet_content = tokenizeText(tweet_content)

            # Getting tweet classification
            tweet_class = row[2]

            # Running test algorithm 
            predicted_class = testNaiveBayes(tweet_content, predicted_class_information, vocab_size)
            
            #Calculating accruancy
            if predicted_class in tweet_class:
                accuracy += 1
            
    print(accuracy)
    print(total_tweets)    
    print(accuracy/total_tweets)

   