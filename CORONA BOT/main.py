import nltk
import numpy
import tflearn
import tensorflow
import pickle
import random
import json
nltk.download('punkt')

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#loading the json data
with open("WHO.json") as file:				 
	data = json.load(file)
	
#print(data["intents"])
try:
	with open("data.pickle", "rb") as f:
		words, l, training, output = pickle.load(f)
except:
	
	# Extracting Data
	words = []
	l = []
	docs_x = []
	docs_y = []
	
# converting each pattern into list of words using nltk.word_tokenizer 
	for i in data["intents"]: 
		for p in i["patterns"]:
			wrds = nltk.word_tokenize(p)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(i["tag"])

			if i["tag"] not in l:
				l.append(i["tag"])
	# Word Stemming		 
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]		 
	words = sorted(list(set(words)))
	l = sorted(l)									 
	
	# This code will simply create a unique list of stemmed 
	# words to use in the next step of our data preprocessing
	training = []
	output = []
	out_empty = [0 for _ in range(len(l))]
	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row = out_empty[:]
		output_row[l.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)
		
	# Finally we will convert our training data and output to numpy arrays 
	training = numpy.array(training)	 
	output = numpy.array(output)
	with open("data.pickle", "wb") as f:
		pickle.dump((words, l, training, output), f)

		
# Developing a Model	 
tensorflow.reset_default_graph()				 

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)


# remove comment to not train model after you satisfied with the accuracy
model = tflearn.DNN(net)
"""try:							 
	model.load("model.tflearn")
except:"""

# Training & Saving the Model
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)	 
model.save("model.tflearn")

# making predictions
def bag_of_words(s, words):							 
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)


def chat():
	print("""Start talking with the bot and ask your
	queries about Corona-virus(type quit to stop)!""")
	
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)])[0]
		results_index = numpy.argmax(results)
		
		#print(results_index)
		tag = l[results_index]
		if results[results_index] > 0.7:
			for tg in data["intents"]:
				if tg['tag'] == tag:
					responses = tg['responses']

			print(random.choice(responses))
		else:
			print("I am sorry but I can't understand")

chat()
