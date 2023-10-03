import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class Trainer_class:
    model = object()
    
    def build_model():
        # Getting the dataset
        sonar_data = pd.read_csv('sonar_data.csv', header=None)
        # separating data and Labels
        X = sonar_data.drop(columns=60, axis=1)
        Y = sonar_data[60]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
        Trainer_class.model = LogisticRegression()
        Trainer_class.model.fit(X_train, Y_train)
        #accuracy on training data
        X_train_prediction = Trainer_class.model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
        X_test_prediction = Trainer_class.model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 

    def get_prediction(input_data):
        input_data_as_numpy_array = np.asarray(input_data)
        # reshape the np array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = Trainer_class.model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0]=='R'):
            return {"status":200,"message":'The object is a Rock',"prediction":"R"}
        else:
            return {"status":200,"message":'The object is a mine',"prediction":"M"}
        
        