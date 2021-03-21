from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam



def MLP(input_dim, num_classes, hidden_layer_sizes = (100,), activation='relu', solver='sgd', alpha = 1e-4, momentum = 0.2, learning_rate_init=1e-4):
    """
    Fully Connected Feed Forward Neural Network Model for Classification.
    
    :param input_dim: input dimension (dimension of the problem i.e. data)
    :param num_classes: number of classes.
    :param hidden_layers: hidden layers given as a tuple.
    :param activation: non-linearity to use.
    :param solver: optimisation algorihtm to use.
    :param alpha: parameter of l2 penalty to apply on bias and output.
    """
    # create model
    model = Sequential()
    
    # first layer
    model.add(Dense(hidden_layer_sizes[0], 
                    input_dim=input_dim, 
                    activation=activation, 
                    bias_regularizer=regularizers.l2(alpha),
                    activity_regularizer=regularizers.l2(alpha)))
    model.add(Dropout(0.2))
    
    # unroll ffn
    # 1 since we already created the first layer
    for i in range(1, len(hidden_layer_sizes)):
        model.add(Dense(hidden_layer_sizes[i], 
                        activation=activation, 
                        bias_regularizer=regularizers.l2(alpha),
                        activity_regularizer=regularizers.l2(alpha)))
        
    # last layer
    model.add(Dense(num_classes, activation='softmax'))
    
    # Optimiser and compile model 
    if solver=='sgd':
        optimizer = SGD(lr=learning_rate_init)
    elif solver=='adam':
        optimizer = Adam(lr=learning_rate_init)
    else :
        optimizer = solver
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model