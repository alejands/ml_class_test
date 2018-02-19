#! /usr/bin/python
'''
classifier_test.py
this is a simple script to test if one can treat a classifier as an observable
quantity. 

Alejandro Sanchez
Carnegie Mellon University
Feb 9, 2018
'''
from __future__ import division, print_function
import sys, csv
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

<<<<<<< HEAD
###paramters###-----------------------------------------------------------------
=======
###paramters###
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af
train_file = 'mnist_train.csv'
test_file = 'mnist_test.csv'
net_size = (100,)					#sklearn default set to (100,)
signal = 9							#digit we are treating as signal
<<<<<<< HEAD
fig_name = 'class_test_dig_{0}.png'.format(signal)

###optional flags###
plot_log_class = True				#default: True
see_full_bg = False					#default: False		include bkg in hist
see_full_sig = False				#default: False		include sig in hist
log_hist = False					#default: False
=======
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af

###main program###--------------------------------------------------------------
def main():

	#grab data from csv files
	print('reading data...')
	train_labels, train_inputs = getData(train_file)
	test_labels, test_inputs = getData(test_file)

<<<<<<< HEAD
	#setup and train classifier
=======
	#setup and train
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af
	train_signals = getSignals(train_labels, signal)
	test_signals = getSignals(train_labels, signal)

	print('initializing classifier...')
	net = MLPClassifier(hidden_layer_sizes=net_size ,solver='adam', verbose=True)
	net.fit(train_inputs, train_signals)

	#testing classifier
<<<<<<< HEAD
	if plot_log_class:
		predictions = net.predict_log_proba(test_inputs)
	else:
		predictions = net.predict_proba(test_inputs)

=======
	predictions = net.predict_log_proba(test_inputs)
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af
	predictions = [y for x,y in predictions]
	pred_per_digit = []
	dig_labels = []

	#separating classifier outputs into real values
	for digit in range(10):
<<<<<<< HEAD
		if digit is signal and not see_full_sig:	#cut out signal to see better
=======
		if digit is signal:			#cut out signal to see better
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af
			continue
		dig_labels.append(str(digit))
		digit_predictions = []
		for i in range(len(test_labels)):
<<<<<<< HEAD
			if test_labels[i] is digit: 
				if predictions[i] != float('-inf'):
					digit_predictions.append(predictions[i])
				elif see_full_bg:					#assign a value to -inf (optional)
					digit_predictions.append(-40.0)
		pred_per_digit.append(digit_predictions)

	#plot histograms separated by real digit values
	if plot_log_class:
		title = 'log(classifier)'
	else:
		title = 'classifier'

	plt.hist(pred_per_digit, label=dig_labels, log=log_hist, bins=30, histtype='barstacked')
	plt.title('{0} for {1}'.format(title, signal))
	plt.xlabel(title)
	plt.legend()
	plt.savefig(fig_name)
	print('Figure saved in {0}'.format(fig_name))
=======
			if test_labels[i] is digit and predictions[i] != float('-inf'):
				digit_predictions.append(predictions[i])
		pred_per_digit.append(digit_predictions)

	#plot histograms separated by real digit values
	plt.hist(pred_per_digit, label=dig_labels, log=False, bins=30, histtype='barstacked')
	plt.title('log(classifier) for {0}'.format(signal))
	plt.xlabel('log(classifier)')
	plt.legend()
	plt.savefig('class_test_dig_{0}_deep.png'.format(signal))
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af
	plt.show()

###end main###

###function definitions###------------------------------------------------------
def getData(filename):
	'''fetches data and returns list of inputs and filenames'''
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		data = list(reader)

	labels, inputs = [], []

	for entry in data:
		label = int(entry[0])
		inpt = map(int,entry[1:])

		labels.append(label)
		inputs.append(inpt)
		
	return labels, inputs

def getSignals(labels, n):
	'''transforms labels into classifier for digit n'''
	signals = [0]*len(labels)
	for i in range(len(labels)):
		if labels[i] is n:
			signals[i] = 1

	return signals

#-------------------------------------------------------------------------------
try:
	main()
except KeyboardInterrupt:
<<<<<<< HEAD
	print('\n\n\tded.\n')
=======
	print('\n\tded.\n')
>>>>>>> e17a82dc5792ab24595bcfc29faf04472b4264af
	sys.exit(0)
