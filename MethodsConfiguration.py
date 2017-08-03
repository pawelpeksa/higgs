import json

class MethodsConfiguration:
    def __init__(self):
        self.svm = SVM()
        self.ann = ANN()
        self.random_forest = RandomForest()
        self.decision_tree = DecisionTree()

    def toDict(self):
        jsonObj = dict()
        jsonObj['svm'] = self.svm.__dict__
        jsonObj['ann'] = self.ann.__dict__
        jsonObj['random_forest'] = self.random_forest.__dict__
        jsonObj['decision_tree'] = self.decision_tree.__dict__

        return jsonObj

    def save(self, filepath):
	 with open(file_name, filepath) as output:
		 json.dump(config.toDict(), output) 


#TODO: change namges of the class below to have suffix Config
class SVM:
    def __init__(self):
        self.C = 0.1


class ANN:
    def __init__(self):
        self.hidden_neurons = 1
        self.solver = 'adam'
        self.alpha = 0.1


class RandomForest:
    def __init__(self):
        self.max_depth = 1
        self.n_estimators = 1


class DecisionTree:
    def __init__(self):
        self.max_depth = 1
