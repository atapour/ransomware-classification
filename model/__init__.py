from model.the_model import TheModel

def create_model(args, weights, classes):
    """
    Create, initialize and return the model

    Parameters
    ----------
    weights: weights needed for class balancing
    classes: the name of the classes

    Returns
    ----------
    model: the model used for training or testing
    """

    model = TheModel()
    model.initialize(args, weights, classes)
    print("The model has been created.")

    return model
