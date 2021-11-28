class Dataset:
    
    def __init__(self, x_train,y_train,x_test,y_test,x_val,y_val,y_test_not_categorical):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val
        self.y_test_not_categorical = y_test_not_categorical