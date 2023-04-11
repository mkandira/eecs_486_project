import sys, os
from preprocess import tokenizeText, removeStopwords, stemWords
from porter import PorterStemmer
import string
import emoji
import csv

    # def is_emoji(s):
    #     return s in emoji.UNICODE_EMOJI

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
            # Opening the file
            tweet_content = row[1]
            
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

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    # >>> print(emoji.demojize('Python is 👍'))
                    # Python is :thumbs_up: 

                    
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

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
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

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
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

                    # Convert all emojis to plaintext
                    word = emoji.demojize(word)
                    
                    # Adding to total docs
                    if word not in vocab_set and word not in string.punctuation:
                        vocab_set.append(word)

                    
                    if word in predicted_class_information["boomer"]["tf"] and word not in string.punctuation:
                        predicted_class_information["boomer"]["tf"][word] += 1
                    elif word not in string.punctuation:
                        predicted_class_information["boomer"]["tf"][word] = 1
    
    return predicted_class_information, len(vocab_set)

            
# input: the file path to be used for test;
# input: the output produced by trainNaiveBayes;
# output: predicted class (the string “true” or the string “fake”. You can assume these to be the only classes to be predicted
def testNaiveBayes(input_tweet: str, predicted_class: dict, vocab_size: int) -> str:
    """Take in class probabilites and total vocab size and test on input tweet."""

    # build list of probabilities based on the token
    total_docs = predicted_class["gen_z"]["num_docs"] + predicted_class["millenial"]["num_docs"] + predicted_class["gen_x"]["num_docs"] + predicted_class["boomer"]["num_docs"]
    gen_z_probability = predicted_class["gen_z"]["num_docs"]/total_docs
    gen_x_probability =  predicted_class["gen_x"]["num_docs"]/total_docs
    millenial_probability =  predicted_class["millenial"]["num_docs"]/total_docs
    boomer_probability =  predicted_class["boomer"]["num_docs"]/total_docs

    # loop through tokens and get probability for fake and true
    for token in input_tweet:
        # Convert all emojis to plaintext
        token = emoji.demojize(token)
        # Adding to gen z probability
        if token not in predicted_class["gen_z"]["tf"]:
            prob = 1 / (predicted_class["gen_z"]["total_words"] + vocab_size)
            gen_z_probability *= float(prob)
        else:
            prob = (predicted_class["gen_z"]["tf"][token] + 1) / \
            float(predicted_class["gen_z"]["total_words"] + vocab_size)
            gen_z_probability *= float(prob)

        # Adding to millenial probability
        if token not in predicted_class["millenial"]["tf"]:
            prob = 1 / float(predicted_class["millenial"]["total_words"] + vocab_size)
            millenial_probability *= float(prob)
        else:
            prob = (predicted_class["millenial"]["tf"][token] + 1) / \
            float(predicted_class["millenial"]["total_words"] + vocab_size)
            millenial_probability *= float(prob)

        # Adding to  gen x probability
        if token not in predicted_class["gen_x"]["tf"]:
            prob = 1 / (predicted_class["gen_x"]["total_words"] + vocab_size)
            gen_x_probability *= float(prob)
        else:
            prob = (predicted_class["gen_x"]["tf"][token] + 1) / \
            float(predicted_class["gen_x"]["total_words"] + vocab_size)
            gen_x_probability *= float(prob)

        # Adding to boomer probability
        if token not in predicted_class["boomer"]["tf"]:
            prob = 1 / (predicted_class["boomer"]["total_words"] + vocab_size)
            boomer_probability *= float(prob)
        else:
            prob = (predicted_class["boomer"]["tf"][token] + 1) / \
            float(predicted_class["boomer"]["total_words"] + vocab_size)
            boomer_probability *= float(prob)


   
    maxProb = max(gen_z_probability,gen_x_probability,millenial_probability,boomer_probability)
    if gen_z_probability == maxProb: 
        return 'gen_z'
    elif gen_x_probability == maxProb:  
        return 'gen_x' 
    elif millenial_probability == maxProb:
        return 'millenial'
    else:
        return 'boomer'
    

if __name__ == "__main__":
    # Getting the filename from the command line
    test_file = os.fsdecode(sys.argv[1])
    training_file = os.fsdecode(sys.argv[2])

    # For each class we need: number of documents in that class, number of words in that class, word/count dictionary for each class
    # Size of vocab
    # predicted_class_information => key = class_name, value = {"num_docs": 5, "total_words": 1000, "tf": {}}
    predicted_class_information, vocab_size = trainNaiveBayes(training_file)
  
        
    #mTesting the Naive Bayes classifier:
    accuracy = 0
    total_tweets = 0
    with open(test_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        total_tweets += 1
        for row in reader:
            tweet_content = row[1]

            # Tokenizing text
            tweet_content = tokenizeText(tweet_content)

            #Getting filename
            tweet_class = row[2]
    
            predicted_class = testNaiveBayes(tweet_content, predicted_class_information, vocab_size)
            print(predicted_class)
            print(tweet_class)
            print("-----")
            #Calculating accruancy
            if predicted_class == tweet_class:
                accuracy += 1
        
    print(accuracy/total_tweets)

   