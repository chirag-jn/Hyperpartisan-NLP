import pickle

def loadPickle(filename):
    file = open(filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model

def savePickle(model, filename):
    file = open(filename, 'wb+')
    pickle.dump(model, file)
    file.close()

article_training_data_loc = 'Data/articles-training-byarticle-20181122.xml'
article_ground_truth_data_loc = 'Data/ground-truth-training-byarticle-20181122.xml'
training_data_schema = 'Data/article.xsd'
ground_truth_schema = 'Data/ground-truth.xsd'

