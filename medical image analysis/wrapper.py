class Wrapper(estimator):
    def __init__(self, param='param'):
        self.param = param
        
        
    def fit(self):
        return self
    

    def predict(self, X_new):
        self.model.predict(X_new)


    def predict_proba(self, X_test, y_test):
        return self.model.predict_proba(X_test, y_test)
    