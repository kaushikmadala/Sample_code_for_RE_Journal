#import tensorflow as tf
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Activation, Dropout, Conv1D, MaxPooling1D, Conv2D, Convolution1D
from keras.layers import LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#from keras.layers.embeddings import Embedding
from keras.layers import Embedding
from keras.preprocessing import sequence
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report
from talos.metrics.keras_metrics import fmeasure_acc, recall_acc, precision_acc, matthews_correlation_acc


#Doclevel
#f = open('train4.txt','r')
#valfile = open('validation4.txt','r')
#testfile = open('test4.txt','r')
#fp_test = open('test4_entire_data_results_gru.txt','w')

#Withindoc for entire data
f = open('train9.txt','r')
valfile = open('validation9.txt','r')
testfile = open('test9.txt','r')
fp_test = open('Withindoc_test9_entire_data_results_gru.txt','w')
num_epochs = 48

inputdata=[]
training_labels=[]
embeddings = defaultdict(list)
wordindexlist = defaultdict(int)

itval = 0
sequences=[]
sequence_labels=[]
maxlen = 0
count1 = 0
inputdata=[]
training_labels=[]


for line in f:
if (len(line.strip())!= 0) and (len(line.split(" "))>=1):
words = line.split(" ")
words[0] = words[0].strip("\n\r")
#print "word zero is ",words[0] #," word one is ",words[2]
if "BOS" == words[0]:
if itval == 0:
pass
else:
datatoberemoved = inputdata
labeldatatoberemoved = training_labels
inputdata = filter(lambda x: x not in datatoberemoved,inputdata)
training_labels = filter(lambda x: x not in labeldatatoberemoved,training_labels)

count1 = 0
elif "EOS" == words[0]:

sequences.append(inputdata)
sequence_labels.append(training_labels)
itval+=1

if maxlen <= count1:
maxlen = count1

else:
# print "count is",count
#
# MODEL ELEMENTS      |  NUMBER labels
#---------------------------------------------------------
# Component | 1
# State | 2
# Actor | 3
# Transition Condition | 4
#
################################################################
inputdata.append(words[0])
#print words[0]
if "TC" in words[2] :
# print "equal to B"
label = [0,0,0,0,1]
training_labels.append(label)
count1+=1
elif "S" in words[2] :
label = [0,0,1,0,0]
training_labels.append(label)
count1+=1
elif "A" in words[2] :
label = [0,0,0,1,0]
training_labels.append(label)
count1+=1
elif "C" in words[2] :
label = [0,1,0,0,0]
training_labels.append(label)
count1+=1
else:
# print "equal to O"
label = [1,0,0,0,0]
training_labels.append(label)
count1+=1
f.close()



# VALIDATION DATA

val_sequences=[]
val_sequence_labels=[]
valinput=[]
vallabels=[]
itval = 0
for line in valfile:
if len(line.strip())!= 0 and (len(line.split(" "))>=1):
words = line.split(" ");
words[0] = words[0].strip("\n\r")
if "BOS" == words[0]:
if itval == 0:
pass
else:
datatoberemoved = valinput
labeldatatoberemoved = vallabels
valinput = filter(lambda x: x not in datatoberemoved,valinput)
vallabels = filter(lambda x: x not in labeldatatoberemoved,vallabels)
count1 = 0

elif "EOS" == words[0]:

val_sequences.append(valinput)
val_sequence_labels.append(vallabels)
if maxlen <= count1:
maxlen = count1
else:
itval = 1;
valinput.append(words[0])
if "TC" in words[2]:
label = [0,0,0,0,1]
vallabels.append(label)
elif "S" in words[2]:
label = [0,0,1,0,0]
vallabels.append(label)
elif "A" in words[2]:
label = [0,0,0,1,0]
vallabels.append(label)
elif "C" in words[2]:
label = [0,1,0,0,0]
vallabels.append(label)
else:
label = [1,0,0,0,0]
vallabels.append(label)
count1 = count1+0
valfile.close()


# TEST DATA

test_sequences=[]
test_sequence_labels=[]
testinput=[]
testlabels=[]
itval = 0
for line in testfile:
if len(line.strip())!= 0 and (len(line.split(" "))>=1):
words = line.split(" ");
words[0] = words[0].strip("\n\r")

if "BOS" == words[0]:
if itval == 0:
pass
else:
datatoberemoved = testinput
labeldatatoberemoved = testlabels
testinput = filter(lambda x: x not in datatoberemoved,testinput)
testlabels = filter(lambda x: x not in labeldatatoberemoved,testlabels)
count1 = 0

elif "EOS" == words[0]:
test_sequences.append(testinput)
test_sequence_labels.append(testlabels)
if maxlen <= count1:
maxlen = count1
else:
itval = 1;
testinput.append(words[0])
if "TC" in words[2]:
label = [0,0,0,0,1]
testlabels.append(label)
elif "S" in words[2]:
label = [0,0,1,0,0]
testlabels.append(label)
elif "A" in words[2]:
label = [0,0,0,1,0]
testlabels.append(label)
elif "C" in words[2]:
label = [0,1,0,0,0]
testlabels.append(label)
else:
label = [1,0,0,0,0]
testlabels.append(label)
count1 = count1+1
testfile.close()


embedfile = open('glove_vectors_data.txt','r')

index = 1
for line in embedfile:
#print('line is',line)
words = line.split()
key = words[0]
wordindexlist[words[0]] = index
index = index+1
values=[]
floatvalues=[]
values = words[1:]
floatvalues= np.asarray(values,dtype='float32')
embeddings[key]=floatvalues
embedfile.close()
#embeddings =  embedfile.readlines()
print "size of dict", len(embeddings)
#print len(embeddings["the"])
vocabulary_size = len(embeddings)
embedding_dimensions = len(values)


one_vector = []
for i in range(embedding_dimensions):
one_vector.append(1.0)

key = "$$$"
wordindexlist[key] = 0
embeddings[key] = np.asarray(one_vector,dtype='float32')

embedding_matrix = np.zeros((len(wordindexlist) + 1, embedding_dimensions))
for word, i in wordindexlist.items():
embedding_vector = embeddings.get(word)
if embedding_vector is not None:
embedding_matrix[i] = embedding_vector



f = 1

seq_data_index = []
seq_val_index = []
seq_test_index = []


inputdataindex = []
for seq_id in sequences:
for word_id in seq_id:
if wordindexlist.has_key(word_id):
inputdataindex.append(wordindexlist[word_id])
elif wordindexlist.has_key(word_id.lower()):
inputdataindex.append(wordindexlist[word_id.lower()])
else:
inputdataindex.append(wordindexlist["unk"])
print "unknown word", word_id
seq_data_index.append(inputdataindex)
datatoberemoved = inputdataindex
inputdataindex = filter(lambda x: x not in datatoberemoved,inputdataindex)


valdataindex = []
for tseq_id in val_sequences:
for tword_id in tseq_id:
if wordindexlist.has_key(tword_id):
valdataindex.append(wordindexlist[tword_id])
elif wordindexlist.has_key(tword_id.lower()):
valdataindex.append(wordindexlist[tword_id.lower()])
else:
valdataindex.append(wordindexlist["unk"])
print "unknown word", tword_id
seq_val_index.append(valdataindex)
datatoberemoved = valdataindex
valdataindex = filter(lambda x: x not in datatoberemoved,valdataindex)






testdataindex = []
for tseq_id in test_sequences:
for tword_id in tseq_id:
if wordindexlist.has_key(tword_id):
testdataindex.append(wordindexlist[tword_id])
elif wordindexlist.has_key(tword_id.lower()):
testdataindex.append(wordindexlist[tword_id.lower()])
else:
testdataindex.append(wordindexlist["unk"])
print "unknown word", tword_id
seq_test_index.append(testdataindex)
datatoberemoved = testdataindex
testdataindex = filter(lambda x: x not in datatoberemoved,testdataindex)

testsize = len(seq_test_index)
valsize = len(seq_val_index)
trainsize = len(seq_data_index)
print "test size is ",testsize
print "train size is ",trainsize
print "maximum sequence length is ",maxlen




X1 = pad_sequences(seq_data_index,maxlen=maxlen)
X2 = pad_sequences(seq_val_index,maxlen=maxlen)
X3 = pad_sequences(seq_test_index,maxlen=maxlen)
Y1 = pad_sequences(sequence_labels,maxlen=maxlen)
Y2 = pad_sequences(val_sequence_labels,maxlen=maxlen)
Y3 = pad_sequences(test_sequence_labels,maxlen=maxlen)


#Creating Zero Vector
zero_vector = []
for i in range(embedding_dimensions):
zero_vector.append(0.0)

#Creating a zero label
zero_label = []
for i in range(3):
zero_label.append(0.0)



np.random.seed(7)
# tanh tanh relu softmax
model = Sequential()
#model.add(embeddinglayer)
model.add(Embedding(vocabulary_size+2,embedding_dimensions,weights = [embedding_matrix],input_length=maxlen))
model.add(GRU(128, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(Activation('tanh'))
model.add(GRU(72, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(Activation('tanh'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


#adam = optimizers.Adam(lr=0.001)

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[fmeasure_acc, matthews_correlation_acc])

print('Train...')


checkpointer = ModelCheckpoint(filepath="lstm_size_20_epochs_40.hdf5", verbose=0, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_fmeasure_acc', patience=50)
tbCallBack = TensorBoard(log_dir='log_dir_lstm', histogram_freq=0,
                            write_graph=True, write_images=True)
history = model.fit(X1,Y1,validation_data=[X2,Y2],batch_size=32,epochs=num_epochs,callbacks=[checkpointer,earlystopping,tbCallBack])
preds1 = model.predict(X2,batch_size=10,verbose= 0)
preds1_classes = model.predict_classes(X2,verbose=0)
preds = model.predict(X3,batch_size=10,verbose= 0)
pred_classes = model.predict_classes(X3,verbose = 0)


val_actual = []
val_pred = []

val1 = testsize
val2 = maxlen
val3 = valsize
for j in range(val3):
for k in range(val2):
if(X2[j][k]!=0):
val_actual.append(np.argmax(Y2[j][k]))
val_pred.append(preds1_classes[j][k])


test_actual = []
test_pred = []
for j in range(val1):
for k in range(val2):
if(X3[j][k]!=0):
test_actual.append(np.argmax(Y3[j][k]))
test_pred.append(pred_classes[j][k])

# CONFUSION MATRIX
print ("Validation Confusion matrix:")
cf_val = confusion_matrix(val_actual,val_pred)
print cf_val

# for validation data
metric_values_for_val = precision_recall_fscore_support(val_actual,val_pred)


#print ("Test Confusion matrix:")
cf_test = confusion_matrix(test_actual,test_pred)

# for test data)
metric_values = precision_recall_fscore_support(test_actual,test_pred)

print ("shape is ",pred_classes.shape)
print ("shape of preds is ",preds.shape)

print('Validation RESULTS:')
print('Test Precision:', metric_values_for_val[0])
print('Test Recall:', metric_values_for_val[1])
print('Test Macro F1-score:', metric_values_for_val[2])
print('Test Support:', metric_values_for_val[3])


print('Test RESULTS:')
print('Test Precision:', metric_values[0])
print('Test Recall:', metric_values[1])
print('Test Macro F1-score:', metric_values[2])
print('Test Support:', metric_values[3])

print (history.history['loss'])
print (history.history['val_loss'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#fp_test = open('Document_level_results/All_in_one_results/test_results_gru_fold3.txt','w')
val1 = testsize
val2 = maxlen
fp_test.write(classification_report(val_actual, val_pred, target_names=["0","1","2","3","4"]))
fp_test.write("*******************************************************************************************************")
fp_test.write("*******************************************************************************************************")
fp_test.write(classification_report(test_actual, test_pred, target_names=["0","1","2","3","4"]))
fp_test.write("*******************************************************************************************************")
fp_test.write("*******************************************************************************************************")
val1 = testsize
val2 = maxlen
val3 = valsize
fp_test.write("*******************************************************************************************************")
fp_test.write("*******************************************************************************************************")
fp_test.write(" \n VALIDATION RESULTS \n")
for j in range(val3):
# val2 = len(test_sequences[j])
for k in range(val2):
fp_test.write(wordindexlist.keys()[wordindexlist.values().index(X2[j][k])])
fp_test.write("\t")
fp_test.write('%d' % preds1_classes[j][k])
fp_test.write("\t")

mx_val = np.argmax(Y2[j][k])

fp_test.write('%d' % mx_val)
fp_test.write("\n")

fp_test.write(" \n TEST RESULTS \n")
for j in range(val1):
# val2 = len(test_sequences[j])
for k in range(val2):
fp_test.write(wordindexlist.keys()[wordindexlist.values().index(X3[j][k])])
fp_test.write("\t")
fp_test.write('%d' % pred_classes[j][k])
fp_test.write("\t")

mx_val = np.argmax(Y3[j][k])

fp_test.write('%d' % mx_val)
fp_test.write("\n")
#fp.write(training_labels[j])
#fp.write(line)
