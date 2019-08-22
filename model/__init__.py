from model.the_model import TheModel

def create_model(args, weights, classes):

    model = TheModel()
    model.initialize(args, weights, classes)
    print("The model has been created.")

    return model
