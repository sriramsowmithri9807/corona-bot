            #CORONA HELP BOT
This is a chatbot that will give answers to most of your corona-related questions/FAQ. The chatbot will give you answers from the data given by WHO(https://www.who.int/). This will help those who need information or help to know more about this virus.

It uses a neural network with two hidden layers(enough for these QnA) that predicts which pattern matches the user’s question and sends it towards that node. More patterns can be added to the user’s questions to train it for more improved results and add more info about coronavirus in the JSON file. The more you train this chatbot the more it gets precise. The advantage of using deep learning is that you don’t have to ask the same question as written in the JSON file cause stemmed words from the pattern are matched with the user question

Prerequisites: 

Python 3
NumPy
nltk
TensorFlow v.1.15 (no GPU required)
tflearn

Training Data: 
To feed the data to the chatbot I have used JSON with possible question patterns and our desired answers. 
The JSON file used for the project is WHO 
For this project, I have named my JSON file WHO.json 
The JSON file tag is the category in which all those responses fall. 
patterns are used for listing all possible question patterns. 
responses contain all the responses with respect to the patterned questions 

Bag of Words: 
As we know neural networks and machine learning algorithms require numerical input. So our list of strings won’t cut it. We need some way to represent our sentences with numbers and this is where a bag of words comes in. What we are going to do is represent each sentence with a list of the length of the number of words in our model vocabulary. Each position in the list will represent a word from our vocabulary the position in the list is a 1 then that will mean that the word exists in our sentence, if it is a 0 then the word is not present.

Developing a Model:
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) 
model.save("model.tflearn")
Our chatbot will predict the answers based on the model we train. Here we will use the neural networks to build the model. The goal of our network will be to look at a bag of words and give a class that they belong to (one of our tags from the JSON file).

Input: What is coronavirus?
Output: COVID-19 is an infectious disease caused by the most recently discovered coronavirus. This new virus and disease were unknown before the outbreak began in Wuhan, China, in December 2019.
Input: What are the symptoms of COVID 19?
Output: The most common symptoms of COVID-19 are fever, tiredness, and dry cough. Some patients may have aches and pains, nasal congestion, runny nose, sore throat, or diarrhoea. These symptoms are usually mild and begin gradually. Some people become infected but don’t develop any symptoms and don't feel unwell. Most people (about 80%) recover from the disease without needing special treatment.


              FOR MORE INFORMATION 
              MAILID :sowmithrisriram7@gmail.com
