import sys, re, os, math
from preprocess import removeSGML, tokenizeText, removeStopwords, stemWords
from porter import PorterStemmer
import glob
import string

def trainNaiveBayes(training_file_paths):
    # Initialzing vocab_set
    vocab_set = []
    true_vocab_set = {} # key = word, value = occurences
    fake_vocab_set = {}  # key = word, value = occurences
    num_true_docs = 0
    num_fake_docs = 0
    total_words_true = 0
    total_words_fake = 0

    for path in training_file_paths:
        # Opening the file
        file_content = open(path, "r").read()
        
        # Removing the SGML tags
        file_content = removeSGML(file_content)

        # Tokenizing text
        file_content = tokenizeText(file_content)

        #Getting filename
        file_class = path.split("/")[-1][:4]
            
        # If file is in class "true"
        if file_class == "true":
            # Incrementing counts of true documents
            num_true_docs += 1

            # Looping through file content
            for word in file_content:
                total_words_true += 1
                word = word.lower()
                # Adding to total docs
                if word not in vocab_set and word not in string.punctuation:
                    vocab_set.append(word)
                
                if word in true_vocab_set and word not in string.punctuation:
                    true_vocab_set[word] += 1
                elif word not in string.punctuation:
                    true_vocab_set[word] = 1
        # If file is fake
        elif file_class == "fake":
            # Incrementing counts of true documents
            num_fake_docs += 1

            # Looping through file content
            for word in file_content:
                # Adding to total docs
                total_words_fake += 1
                word = word.lower()
                if word not in vocab_set and word not in string.punctuation:
                    vocab_set.append(word)

                if word in fake_vocab_set and word not in string.punctuation:
                    fake_vocab_set[word] += 1
                elif word not in string.punctuation:
                    fake_vocab_set[word] = 1
    
    return num_true_docs, num_fake_docs, true_vocab_set, fake_vocab_set, total_words_true, total_words_fake, vocab_set

            

def testNaiveBayes(test_file_path,num_true_docs, num_fake_docs, true_vocab_set, fake_vocab_set, total_words_true, total_words_fake, vocab_set):
    # Opening the file
    file_content = open(test_file_path, "r").read()
    
    # Removing the SGML tags
    file_content = removeSGML(file_content)

    # Tokenizing text
    file_content = tokenizeText(file_content)

    # Assuming class is true
    p_true = num_true_docs/num_true_docs+num_fake_docs

    # Looping through words in content
    for word in file_content:
        n_k = 0
        if word in true_vocab_set:
            n_k = true_vocab_set[word]
        p_true *= ((n_k + 1)/(total_words_true+len(vocab_set)))

    # Assuming class is false
    p_false = num_true_docs/num_true_docs+num_fake_docs

    # Looping through words in content
    for word in file_content:
        n_k = 0
        if word in fake_vocab_set:
            n_k = fake_vocab_set[word]
        p_false *= ((n_k + 1)/(total_words_fake+len(vocab_set)))
    
    # If the probability of word being class true is greater than false, return true
    if p_true > p_false:
        return "true"
    # Else return false
    return "fake"

if __name__ == "__main__":
    # Getting the filename from the command line
    folder = os.fsdecode(sys.argv[1])

    # Initializing list of files
    filename_paths = []
    filename_contents = {}

    #Outputting information to file
    output_file = open('naivebayes.output.' + folder[:len(folder)-1], 'w')

    for filename in glob.glob(folder + '*'):
        # Getting file name and appending its path to list
        filename_paths.append(str(filename))
        content = open(filename, "r").read()
        filename_contents[filename] = content 
    
    accuracy = 0
    chunk_size = math.ceil(len(filename_paths)/10)
    chunks = [filename_paths[x:x+chunk_size] for x in range(0, len(filename_paths), chunk_size)]
    files_seen = []

    # Looping through filename paths
    #for test_file_paths in chunks:
    i = 0
    for test_file_paths in chunks:
        # Initializing list with training probabilities
        training_file_paths = list(set(filename_paths) - set(test_file_paths))
       
        # Training a Naive Baye s classifier:
        num_true_docs, num_fake_docs, true_vocab_set, fake_vocab_set, total_words_true, total_words_fake, vocab_set = trainNaiveBayes(training_file_paths)
        true_vocab_set_prob = {}
        fake_vocab_set_prob = {}
        # Code to print words with highest probability
        # if i == 0:
        #     for word in true_vocab_set:
        #         if word not in string.punctuation:
        #             true_vocab_set_prob[word] = (true_vocab_set[word] + 1)/(total_words_true + len(vocab_set))
        #     sorted_true = sorted(true_vocab_set_prob.items(), key=lambda x:x[1])
        #     sorted_true.reverse()
        #     for i in range(0, 10):
        #         print(sorted_true[i][0] + " " + str(sorted_true[i][1]))
            
        #     for word in fake_vocab_set:
        #         if word not in string.punctuation:
        #             fake_vocab_set_prob[word] = (fake_vocab_set[word] + 1)/(total_words_fake + len(vocab_set))
        #     sorted_fake = sorted(fake_vocab_set_prob.items(), key=lambda x:x[1])
        #     sorted_fake.reverse()
        #     print()
        #     for i in range(0, 10):
        #         print(sorted_fake[i][0] + " " + str(sorted_fake[i][1]))
       
        #i += 1
        
        # Testing the Naive Bayes classifier:
        for test_file_path in test_file_paths:
            predicted_class = testNaiveBayes(test_file_path, num_true_docs, num_fake_docs, true_vocab_set, fake_vocab_set, total_words_true, total_words_fake, vocab_set)
            output_file.write(str(test_file_path) + " " + str(predicted_class) + '\n')
        
            #Calculating accruancy
            if predicted_class == test_file_path.split("/")[-1][:4]:
                accuracy += 1
        
    print(accuracy/len(filename_paths))








        


    
