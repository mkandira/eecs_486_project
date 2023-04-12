# Determining Age Generation of Twitter Users via Bayesian Classification
EECS 486 Winter 2023 Final Project: Anish Sundaram and Alyssa Hidalgo and Ritika Shetty and Nicholas Cook and Meghana Kandiraju

***

# Description of Project:
This project uses text classification to analyze a person’s tweets and predict which generation they belong to.
 
Each Twitter user is verified and their age is recorded, and this information is then used to train the machine to make accurate predictions of age based on various uses of the language.

The generations have been focused on Gen Z (11-26), Millennials (27-42), Gen X (43-58), and Boomers (59-77). Looking at slang usage, topics discussed, emojis, and more, the classification will provide insight for many different uses.

Our approach involved using a Naive Bayes Classiifier based on Assignment 4 and we trained and tested our classifier on 70,000 and 72,000 tweets respectively. 

***

# File Tree:
```bash
.
├── README.md
├── emoji
├── naivebayes.py
├── porter.py
├── preprocess.py
├── requirements.txt
├── testing_data.csv
└── training_data.csv
```
***

# Datasets:
We had two datasets for our project, one to train our predictor and another to test how accurate our algorithm was. The two datasets were obtained by scrubbing tweets from Twitter of verified users using a cloud based software program called PhantomBuster. These two datasets are named `testing_data.csv` and `training_data.csv` accordingly.

The training and testing data are structed as following:
```sh
[ID],[Tweet],[Classifiction]
Example: 
ChantelJeffries,Just like that,gen z
oliviagobrien,RT @Shazam: Shazam #Josslyn by @oliviagobrien to unlock an exclusive animated video 💙 https://t.co/nhG0ioG8ut,gen z
```
***

# Install:
To clone the project from the public repository link use the following command:
```
git clone git@github.com:mkandira/eecs_486_project.git
```
```
git clone https://github.com/mkandira/eecs_486_project.git
```


Install necessary packages using "pip install". 
```sh
pip install emoji
```
# Running the Code:
We must first run the classifier on the training set, which can be done using the following command:
```sh
python naivebayes.py training_data.csv
```

To then test the classifier on the testing data run the following:
```sh
python naivebayes.py testing_data.csv
```

***

# Authors:
Anish Sundaram (anishsun) and Alyssa Hidalgo (hidalga) and Ritika Shetty (ritikas) and Nicholas Cook (nickcook)and Meghana Kandiraju (mkandira)

# Acknowledgements:
We would like to thank Professor Rada Mihalcea, Artem Abzaliev, Hustin Cao, Aylin Gunal, and
Nam Ho Koh for their support in EECS 486 during Winter 2023. 


