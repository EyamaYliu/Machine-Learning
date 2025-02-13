import os
import argparse
import sys
import pickle
import models
import numpy as np
from scipy.sparse import csr_matrix

from cs475_types import ClassificationLabel, FeatureVector, Instance, Predictor


def load_data(filename):
    """ Load data.

    Args:
        filename: A string. The path to the data file.

    Returns:
        A tuple, (X, y). X is a compressed sparse row matrix of floats with
        shape [num_examples, num_features]. y is a dense array of ints with
        shape [num_examples].
    """

    X_nonzero_rows, X_nonzero_cols, X_nonzero_values = [], [], []
    y = []
    with open(filename) as reader:
        for example_index, line in enumerate(reader):
            if len(line.strip()) == 0:
                continue

            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")
            y.append(int_label)

            for item in split_line[1:]:
                try:
                    # Features are 1 indexed in the data files, so we need to subtract 1.
                    feature_index = int(item.split(":")[0]) - 1
                except ValueError:
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                if feature_index < 0:
                    raise Exception('Expected feature indices to be 1 indexed, but found index of 0.')
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")

                if value != 0.0:
                    X_nonzero_rows.append(example_index)
                    X_nonzero_cols.append(feature_index)
                    X_nonzero_values.append(value)

    X = csr_matrix((X_nonzero_values, (X_nonzero_rows, X_nonzero_cols)), dtype=np.float)
    y = np.array(y, dtype=np.int)

    return X, y


def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")

    parser.add_argument("--data", type=str, required=True, help="The data to use for training or testing.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--algorithm", type=str, help="The name of the algorithm for training.")

    # TODO This is where you will add new command line options
    parser.add_argument("--num-boosting-iterations", type=int, help="The number of boosting iterations to run.",
                        default=10)
    
    
    args = parser.parse_args()
    check_args(args)

    return args


def check_args(args):
    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified in mode \"train\"")
    else:
        if args.predictions_file is None:
            raise Exception("--algorithm should be specified in mode \"test\"")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")


def train(X,y,iteration, algorithm):
    if algorithm.lower() == 'adaboost':
        model = models.Adaboost()
    else:
        raise Exception('The model given by --model is not yet supported.')

    model.fit(X,y,iteration)
    return model
            
    


def write_predictions(predictor,X, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            label = predictor.predict(X)

            for label in label:
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        X,y = load_data(args.data)

        # Train the model.
        predictor = train(X,y,args.num_boosting_iterations, args.algorithm)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data.
        X,y = load_data(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor,X, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

