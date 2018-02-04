import os
import pandas as pd
import tflearn
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from tflearn.data_utils import to_categorical, pad_sequences

if __name__ == '__main__':
	#[1] Read the data
	train_data = np.genfromtxt('training_data.txt',dtype='str')
	test_data = np.genfromtxt('test_data.txt',dtype='str')
	#train_data = np.loadtxt('training_data.txt',  delimiter=' ')
	#test_data = np.loadtxt('test_data.txt')
	train_labels = train_data[0, :]
	train_stars = train_data[1:, 0]
	train_reviews = train_data[1:, 1:]

	test_labels = test_data[0, :]
	#test_stars = train_data[1:, 0]
	test_reviews = test_data[1:, 0:]

	trainX = train_reviews
	trainY = train_stars
	testX = test_reviews
	#testY = test_stars

	print("trainX = ", trainX.shape)
	print("trainY = ", trainY.shape)
	print("testX = ", testX.shape)


	for i in range(0, len(trainX)):
		for j in range(0, len(trainX[0])):
			trainX[i][j] = int(trainX[i][j])

	for i in range(0, len(testX)):
		for j in range(0, len(testX[0])):
			testX[i][j] = int(testX[i][j])

	for i in range(0, len(trainY)):
			trainY[i] = int(trainY[i])
	#for i in range(0, len(trainY)):
			#testY[i] = int(testY[i])



	trainY = to_categorical(trainY, nb_classes = 2)
	#testY = to_categorical(testY, nb_classes = 2)

	clf = RandomForestClassifier(max_depth=50, min_samples_leaf = 2, random_state=0)
	clf.fit(trainX, trainY)
	#score = clf.score(testX, testY)

	#print("score = ", score)

	predictedY = clf.predict(testX)

	predictions = []
	for i in range(0, len(predictedY)):
		if predictedY[i][0] == 1:
			predictions.append(0)
		else:
			predictions.append(1)
	

	# numWrong = 0
	# for i in range(0, len(predictedY)):
	# 	if predictedY[i] != testY[i]:
	# 		numWrong+=1

	# testError = numWrong/len(testY)

	# print("error = ", testError)

	print(predictedY)
	#file = open("submission.txt","w") 
	for i in range(0, len(predictions)):
		#nums = str(i) + ", " + str(predictedY[i])
		#file.write(nums) 
		print(i+1, ",", predictions[i]) 
	 
	#file.close() 


