import numpy as np
import pandas as pd
from sklearn import datasets as ds
from collections import Counter
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
import scipy as sp



'''DATA WRANGLING FUNCTIONS'''

'''
Splits the data set randomly into training and testing according to train_frac. Also shuffles the examples 
so as to not feed the classifier all examples of one class at a time.
'''
def split(data, train_frac):
    classes = data['class'].unique()
    #split into training and testing randomly
    classes_dfs = [data[data['class'] == i] for i in range(len(classes))]
    train_dfs = [cls.sample(frac = train_frac) for cls in classes_dfs]
    test_dfs = [classes_dfs[i].loc[~classes_dfs[i].index.isin(train_dfs[i].index)] for i in range(len(classes))]

    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    #shuffle the examples in the data set
    train_df = train_df.reindex(np.random.permutation(train_df.index))
    test_df = test_df.reindex(np.random.permutation(test_df.index))

    x_train = train_df.as_matrix(columns = train_df.columns[:-1]).astype(np.float64)
    y_train = np.concatenate(train_df.as_matrix(columns = train_df.columns[-1:])).astype(np.int64)

    x_test = test_df.as_matrix(columns = test_df.columns[:-1]).astype(np.float64)
    y_test = np.concatenate(test_df.as_matrix(columns = test_df.columns[-1:])).astype(np.int64)
    
    return x_train, y_train, x_test, y_test

'''
Puts either the mnist or iris data sets into a pandas dataframe and calls split with parameter test_fraction.
Also standardizes the data, and binarizes and vectorizes the output labels
'''
def preprocess_mldata(dataset_name, test_fraction):
    if dataset_name == 'iris':
        dataset = ds.load_iris()
    elif dataset_name == 'mnist':
        dataset = fetch_mldata('MNIST original')
        
    data = pd.DataFrame(dataset.data)
    data['class'] = dataset.target
    x_train, y_train, x_test, y_test = split(data, test_fraction)
    
    # Standardize the training and test data
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    #binarize and vectorize classes
    y_train = y_bin(y_train.astype(np.int64))
    y_test = y_bin(y_test.astype(np.int64))
    
    return x_train, y_train, x_test, y_test

'''
Creates reduced feature dimensionality training and test datasets using PCA and LDA transformations
'''
def dim_reduce(x_train, y_train, x_test, num_pca):    
    pcaclf = PCA().fit(x_train)
    x_pca = pcaclf.transform(x_train)
    y_1d = np.argmax(y_train, axis = 1) #LDA requires a 1D y array
    ldaclf = LDA().fit(x_train, y_1d)
    x_lda = ldaclf.transform(x_train)
    x_pca_lda = np.concatenate((x_pca[:,0:num_pca], x_lda), axis = 1)
    
    x_test_pca = pcaclf.transform(x_test)
    x_test_lda = ldaclf.transform(x_test)
    x_test_pca_lda = np.concatenate((x_test_pca[:,0:num_pca], x_test_lda), axis = 1)
    return x_pca_lda, x_test_pca_lda


'''
Binarizes and Vectorizes the class labels. Only works when classes are numbers that start at 0.
'''
def y_bin(y_data):
    classes = np.unique(y_data)
    y_data_bin = np.zeros([len(y_data), len(classes)])
    for i in range(len(y_data)):
        y_data_bin[i, y_data[i]] = 1
    return y_data_bin



'''NEURAL NETWORK HELPER FUNCTIONS'''

'''
Loops through layers and initializes node weights to either 0 or random, and biases and activations to 0s
''' 
def initialize(layer_sizes, layer_weights_rand):
    w = []; b = []; a = []
    for l in range(0, len(layer_sizes) - 1):
        if layer_weights_rand[l]:
            w = w + [np.random.normal(scale = 1/np.sqrt(layer_sizes[l]), size = (layer_sizes[l+1], layer_sizes[l]))]
        else:
            w = w + [np.zeros((layer_sizes[l+1], layer_sizes[l]))]
        b = b + [np.zeros(layer_sizes[l+1])]
        a = a + [np.zeros(layer_sizes[l])]
    a = a + [np.zeros(layer_sizes[-1])]    
    return w, b, a

def sigmoid(z_value):
    return sp.special.expit(z_value)

'''
Neural network activation function, calls 'function' activation function
''' 
def act_fn(function, z_value):
    if function == 'sigmoid':
        return sigmoid(z_value)

'''
Forward propagation function. Calculates z-values for each layer in the NN for a single epoch
'''    
def forward_prop(w, b, a):
    num_props = len(w)    
    for prop in range(num_props):
        z_values = np.dot(w[prop], a[prop]) + b[prop]
        a[prop+1] = act_fn('sigmoid', z_values)

'''
Error function
'''                
def err_fn(prediction, target):
    return sum((prediction - target)**2) * 0.5

'''
Calculates the derivative of the activation values with respect to the z-values
'''
def calc_dadz(function, activations):
    if function == 'sigmoid':
        return activations * (1 - activations)

'''
Calculates the derivative of the error with respect to the z-values
'''    
def calc_dedz_output_layer(predictions, target):
    return -(target - predictions) * calc_dadz('sigmoid', predictions)

'''
a = 'activations' are the activation values for the layer from which the weights originate.
'dedz_previous': error derivative with respect to the z values of the layer at which the weights terminate.
'''
def update_weights(w, a, dedz_previous, alpha, reg):        
    if reg != 0:
        return w - alpha * (np.outer(dedz_previous, a) + reg*w)
    else:
        return w - alpha * np.outer(dedz_previous, a)

'''
dedz_previous' is the error derivative with respect to the z values of the layer to which the bias is applied.
'''    
def update_biases(b, dedz_previous, alpha):
    return b - alpha * dedz_previous

def calc_dedz_next(a, w, dedz_previous):
    return np.dot(dedz_previous, w) * calc_dadz('sigmoid', a)

'''
Backpropagation function, updates the weights and biases in the NN for a single epoch
'''
def back_prop(w, b, a, predictions, target, layer_sizes, alpha, reg, skip_layers):
    num_props = len(layer_sizes) - 1
    dedz_previous = calc_dedz_output_layer(predictions, target)
    for l in range(num_props-1, (-1 + skip_layers), -1): #Index starting at the end of the list and moving backwards
        if l != 0:
            dedz_next = np.zeros(layer_sizes[l])
            dedz_next = calc_dedz_next(a[l], w[l], dedz_previous)
        #update weights and biases
        w[l] = update_weights(w[l], a[l], dedz_previous, alpha, reg)
        b[l] = update_biases(b[l], dedz_previous, alpha)
        if l != 0:
            dedz_previous = dedz_next

            
'''ACCURACY FUNCTIONS'''

'''
Binarizes the activation values in the output layer
'''
def bin_output_vals(a_values):
    bin_values = np.zeros([len(a_values), len(a_values[0])])
    if len(a_values[0]) > 1:
        for i in range(len(a_values)):
            bin_values[i, np.argmax(a_values[i])] = 1
    else:
        for i in range(len(a_values)):
            bin_values[i, 0] = np.argmin([abs(a_values[i][0] - 0), abs(a_values[i][0] - 1)])
    return bin_values

'''
Computes prediction accuracy and returns accuracy as well as a confusion matrix
'''
def accuracy(bin_values, y_test):
    correct = 0.
    #Initialize a confusion matrix
    conf_mat = np.zeros([bin_values.shape[1],bin_values.shape[1]])
    for i in range(len(bin_values)):
        conf_mat[np.argmax(y_test[i]), np.argmax(bin_values[i])] += 1
        if np.all(bin_values[i] == y_test[i]):
            correct += 1
    acc = correct / len(y_test)
    return acc, conf_mat   

'''
Returns a validation set of size validation_frac
'''
def val_split(X, Y, validation_frac):
    num_examples = X.shape[0]
    cutoff = np.round(num_examples*validation_frac).astype(np.int64)
    return X[cutoff:num_examples,:], Y[cutoff:num_examples,:], X[0:cutoff,:], Y[0:cutoff,:]




'''MULTILAYER PERCEPTRON FUNCTIONS'''

'''
Returns a single Feed Forward Neural Network

-X: training set
-Y: test set
-layer_sizes: a list of the number of neurons in each layer 
-alpha: the learning rate
-regularization factor
-epochs: the number of training epochs 
-w_children: a list of 2D arrays of the weights of each layer of the children NN's 
-b_children: same for biases
-strip_layers: number of layers to strip from the children NN's, starting from the output layer and back
-skip_layers: number of layers for which to skip backpropagation if we want to keep the weights in certain layers
immutable, starting from the input layer
-layer_weights_rand: list of flags for the layers to be initialized with random weights
-selective_weights: flag where: 
1 = update the virgin crossweights but hold child weights constant (immutable)
0 = all weights are updated (mutable).
-reset_zeros: flag where: 
1 = only update the child weights when we're backpropagating through an earlier layer and 
keep the crossweights at zero when doing backpropagation through certain layers
0 = Everything is mutable
-validation_frac: how much of the data set to use for a validation set
'''
def FFNN(X, Y, layer_sizes, alpha, reg, epochs, w_children, b_children, strip_layers,
         skip_layers, layer_weights_rand, selective_weights, reset_zeros, validation_frac):    
    
    if validation_frac: #create validation set
        X, Y, Xval, Yval = val_split(X, Y, validation_frac)
    
    #initialize arrays for weights, biases, activation and derivatives
    w, b, a = initialize(layer_sizes, layer_weights_rand)
    
    if w_children:
        populate_trained_w_b(w, b, w_children, b_children, strip_layers)        
    if reset_zeros:
        mask = create_zeros_mask(w, w_children, strip_layers)
    
    num_examples = X.shape[0]
    
    #create errors vector
    errors = np.zeros(epochs)
    temp_errors = np.zeros(num_examples)
    
    if validation_frac:
        accuracies = np.zeros(epochs)
        w_max = 0; b_max = 0; epoch_max = 0; acc_max = 0
    
    for epoch in range(epochs):
        for ex in range(num_examples):
            
            #forward propagate
            a[0] = X[ex]
            forward_prop(w, b, a)
            
            #compute and save error
            prediction = a[-1]
            target = Y[ex]
            temp_errors[ex] = err_fn(prediction, target)
            
            #back propagate
            back_prop(w, b, a, prediction, target, layer_sizes, alpha, reg, skip_layers)
            if selective_weights:
                populate_trained_w_b(w, b, w_children, b_children, strip_layers)
            if reset_zeros:
                reset_zeros_fun(w, mask)
            
        errors[epoch] = np.mean(temp_errors)
        if validation_frac:
            accuracies[epoch], _ = FFNN_acc(Xval, Yval, layer_sizes, w, b)
            if (epoch > 0) & (accuracies[epoch] > acc_max):
                w_max = w
                b_max = b
                epoch_max = epoch
                acc_max = accuracies[epoch]
            elif epoch == 0:
                w_max = w
                b_max = b
                acc_max = accuracies[epoch]

    if validation_frac:
        return w_max, b_max, errors, accuracies, epoch_max
    else:
        return w, b, errors, 0, 0


'''
Forward propagates through the NN for a single epoch to return prediction accuracy and a confusion matrix
'''    
def FFNN_acc(x_test, y_test, layer_sizes, w, b):        
    a = [np.zeros(layer_sizes[l]) for l in range(0, len(layer_sizes) - 1)]
    a = a + [np.zeros(layer_sizes[-1])]
    
    num_examples = x_test.shape[0]
    a_values = []
    
    for ex in range(num_examples): #forward propagate
        a[0] = x_test[ex]
        forward_prop(w, b, a)
        a_values.append(a[-1])
    
    bin_values = bin_output_vals(a_values)
    acc, conf_mat = accuracy(bin_values, y_test)
    return acc, conf_mat




'''SYNTHESIS MLP FUNCTIONS'''


'''
Checks that the right data set is being used for either class or feature specialization, so as to avoid getting stuck
trying to generate an enormous number of child NN's by mistake
'''
def check_params(x_train, y_train, specialize_by):
    num_classes = y_train.shape[1]
    num_features = x_train.shape[1]
    if specialize_by == 'class':
        if num_classes > 40:
            raise NameError('Too many classes to spawn class specialized children')
    elif specialize_by == 'features':
        if num_features > 40:
            raise NameError('Too many features to spawn feature specialized children')

'''
Populates the parent network with the weights and biases provided by the children network 
'''
def populate_trained_w_b(w, b, w_children, b_children, strip_layers):
    for l in range(len(w_children[0]) - strip_layers):
        (size_i_child, size_j_child) = w_children[0][l].shape
        (size_i_parent, size_j_parent) = w[l].shape
        if size_i_child == size_i_parent:
            w[l] = np.hstack([w_children[ichild][l] for ichild in range(len(w_children))])
            b[l] = np.mean([b_children[ichild][l] for ichild in range(len(b_children))],axis = 0)
        elif size_j_child == size_j_parent:
            w[l] = np.vstack([w_children[ichild][l] for ichild in range(len(w_children))])
            b[l] = np.concatenate([b_children[ichild][l] for ichild in range(len(b_children))])
        elif size_i_child != size_i_parent & size_j_child != size_j_parent:
            for ichild in range(len(w_children)):
                w[l][size_i_child*ichild:size_i_child*(ichild+1), size_j_child*ichild:size_j_child*(ichild+1)] = \
                    w_children[ichild][l]
                b[l][size_i_child*ichild:size_i_child*(ichild+1)] = b_children[ichild][l]
        else:
            raise NameError('Error: child dimensions match parent dimensions in both i and j')

'''
Creates arrays with zeros where the cross weights are and ones where the child weights are.
Allows to only update the child weights when doing backpropagation
'''            
def create_zeros_mask(w, w_children, strip_layers):
    mask = [0] * (len(w_children[0]) - strip_layers)
    for l in range(len(w_children[0]) - strip_layers):
        (size_i_child, size_j_child) = w_children[0][l].shape
        (size_i_parent, size_j_parent) = w[l].shape
        if size_i_child == size_i_parent:
            mask[l] = np.ones((size_i_parent, size_j_parent))
        elif size_j_child == size_j_parent:
            mask[l] = np.ones((size_i_parent, size_j_parent))
        elif size_i_child != size_i_parent & size_j_child != size_j_parent:
            mask[l] = np.zeros((size_i_parent, size_j_parent))
            for ichild in range(len(w_children)):
                mask[l][size_i_child*ichild:size_i_child*(ichild+1), size_j_child*ichild:size_j_child*(ichild+1)] = \
                    np.ones((size_i_child, size_j_child))
        else:
            raise NameError('Error: child dimensions match parent dimensions in both i and j')
    return mask

'''
Reset the crossweights to their original value of zero
'''
def reset_zeros_fun(w, mask):
    for l in range(len(mask)):
        w[l] = w[l] * mask[l]

'''
When doing class specialization, balances classes so that the number of the examples for the class being trained on
equals the sum of the examples for all of the classes being trained against
'''                
def single_class_balance(x_train, y_train):
    class_rows = np.where(y_train == 1)[0]
    not_class_rows = np.where(y_train == 0)[0]
    subsample_not_class = np.random.choice(not_class_rows, size = len(class_rows), replace=False)
    indices = np.sort(np.concatenate((class_rows, subsample_not_class)))
    x_train_subsample = x_train[indices,:]
    y_train_subsample = y_train[indices,:]
    return x_train_subsample, y_train_subsample

'''
Returns FFNN children specialized in either classes or features, one child NN for each class or each feature

-child_hidden_layers: a list of the the number of nodes for each layer in the children NN's, starting with the input
layer
-specialize_by: 'class' or 'features' depending on whether each child should be trained on a single class or
a single feature in the data set
-balance: whether or not to balance the data set when doing class specialization
'''
def Child_Spawner(x_train, y_train, child_hidden_layers, specialize_by, balance, alpha, reg, epochs, 
                  validation_frac, x_test, y_test):
    check_params(x_train, y_train, specialize_by)
    num_classes = y_train.shape[1]
    num_features = x_train.shape[1]
    
    #specialization by class
    if specialize_by == 'class':
        layer_sizes = [x_train.shape[1]] + child_hidden_layers + [1]
        layer_weights_rand = np.ones([len(layer_sizes)])
        w_children = []; b_children = []; acc_children = []; epoch_acc_children = []; best_epoch_children = []
        #create and trains a child for each class in the data set        
        for iclass in range(num_classes):
            y_train_sub = y_train[:,iclass].reshape([len(y_train),1])
            y_test_sub = y_test[:,iclass].reshape([len(y_test),1])
            if balance:
                x_train_sub, y_train_sub = single_class_balance(x_train, y_train_sub)
            else:
                x_train_sub = x_train
            w, b, errors, epoch_accs, best_epoch = FFNN(x_train_sub, y_train_sub, layer_sizes, alpha, reg, 
                                epochs, [], [], 0, 0, layer_weights_rand, 0, 0, validation_frac)
            acc, _ = FFNN_acc(x_test, y_test_sub, layer_sizes, w, b)
            w_children += [w]
            b_children += [b]
            epoch_acc_children += [[epoch_accs]]
            best_epoch_children += [best_epoch]
            acc_children += [acc]

    #specialization by feature            
    if specialize_by == 'features':
        layer_sizes = [1] + child_hidden_layers + [num_classes]
        layer_weights_rand = np.ones([len(layer_sizes)])        
        w_children = []; b_children = []; acc_children = []; epoch_acc_children = []; best_epoch_children = []
        #create and trains a child for each feature in the data set
        for ifeature in range(num_features):
            w, b, errors, epoch_accs, best_epoch = FFNN(x_train[:,ifeature].reshape([len(x_train),1]), 
            y_train, layer_sizes, alpha, reg, epochs, [], [], 0, 0, layer_weights_rand, 0, 0, validation_frac)
            acc, _ = FFNN_acc(x_test[:,ifeature].reshape([len(x_test),1]), y_test, layer_sizes, w, b)
            w_children += [w]
            b_children += [b]
            epoch_acc_children += [[epoch_accs]]
            best_epoch_children += [best_epoch]
            acc_children += [acc]
    return w_children, b_children, acc_children, epoch_acc_children, best_epoch_children

'''
Builds a parent synthesizer Neural Network from the children weights and biases, and returns accuracy on the test
set, accuracies on the validation set for each epoch, the error for each epoch, and which epoch yielded the best
prediction accuracy on the validation set

-virgin_hidden_layers: a list of the number of neurons for each virgin layer to be added to the parent NN 
'''
def SNN(x_train, y_train, alpha, reg, epochs, w_children, b_children, strip_layers, skip_layers, validation_frac,
    layer_weights_rand, selective_weights, reset_zeros, children_hidden_layers, virgin_hidden_layers, x_test, y_test):
    
    layer_sizes = [x_train.shape[1]] + list(map(lambda x: x*len(w_children), \
            children_hidden_layers[0:len(children_hidden_layers)-strip_layers])) \
            + virgin_hidden_layers + [y_train.shape[1]]

    w, b, errors, accs, best_epoch = FFNN(x_train, y_train, layer_sizes, alpha, reg, epochs, w_children, b_children, 
               strip_layers, skip_layers, layer_weights_rand, selective_weights, reset_zeros, validation_frac)

    acc, conf_mat = FFNN_acc(x_test, y_test, layer_sizes, w, b)
    print('test set accuracy:', acc)
    print('validation test accuracy per epoch:', accs)
    print('Best Epoch:', best_epoch)    
    return acc, errors, accs, best_epoch
