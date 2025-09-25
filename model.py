import pickle

def load_model(model_path):
    # Load the pre-trained machine learning model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
