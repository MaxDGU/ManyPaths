from datasets import MetaModuloDataset, MetaBitConceptsDataset, Omniglot
from models import MLP, CNN, LSTM, Transformer
from constants import *
from generate_concepts import PCFG_DEFAULT_MAX_DEPTH
import os
import torch

def init_misc(experiment, alphabet, num_concept_features_override: int = None):
    if alphabet in ["ancient", "asian"] and experiment == "omniglot":
        alphabet = ALPHABETS[alphabet]
    else:
        alphabet = None
    channels = 3 if experiment == "concept" else 1
    
    # Determine 'bits' (which is n_input for MLP bit tasks)
    if experiment == "concept":
        if num_concept_features_override is not None:
            bits = num_concept_features_override
        else:
            bits = 4  # Default for concept experiment if not overridden
    else:
        bits = 8 # Default for other experiments (e.g., mod with bits data)
        
    n_output = 20 if experiment == "omniglot" else 1
    return alphabet, bits, channels, n_output


def init_dataset(
    experiment, 
    model_arch,
    data_type, 
    skip_param,
    n_support=None, 
    alphabet=None, 
    num_concept_features: int = 8,
    pcfg_max_depth: int = PCFG_DEFAULT_MAX_DEPTH,
    save_train_path: str = None,
    save_val_path: str = None,
    save_test_path: str = None,
    load_train_path: str = None,
    load_val_path: str = None,
    load_test_path: str = None
):
    train_ds, val_ds, test_ds = None, None, None
    generated_train, generated_val, generated_test = False, False, False

    # Attempt to load datasets if paths are provided
    if load_train_path:
        if os.path.exists(load_train_path):
            print(f"Loading train_dataset from {load_train_path}")
            train_ds = torch.load(load_train_path)
        else:
            raise FileNotFoundError(f"Specified load_train_path not found: {load_train_path}")
    
    if load_val_path:
        if os.path.exists(load_val_path):
            print(f"Loading val_dataset from {load_val_path}")
            val_ds = torch.load(load_val_path)
        else:
            raise FileNotFoundError(f"Specified load_val_path not found: {load_val_path}")

    if load_test_path:
        if os.path.exists(load_test_path):
            print(f"Loading test_dataset from {load_test_path}")
            test_ds = torch.load(load_test_path)
        else:
            raise FileNotFoundError(f"Specified load_test_path not found: {load_test_path}")

    # Generate datasets only if not loaded
    if n_support is None: # train_ds is needed
        if train_ds is None:
            print(f"Generating train_dataset for experiment: {experiment}...")
            if experiment == "mod":
                train_ds = MetaModuloDataset(
                    n_tasks=10000,
                    skip=skip_param,
                    train=True,
                    data=data_type,
                    model=model_arch,
                )
            elif experiment == "concept":
                train_ds = MetaBitConceptsDataset(
                    n_tasks=10000,
                    data=data_type,
                    model=model_arch,
                    num_features=num_concept_features,
                    pcfg_max_depth=pcfg_max_depth
                )
            elif experiment == "omniglot":
                train_ds = Omniglot(
                    n_tasks=10000,
                    model=model_arch,
                    train=True,
                    alphabet=alphabet
                )
            else:
                raise ValueError(f"Experiment '{experiment}' for train_dataset generation unrecognized.")
            generated_train = True
            print("Finished generating train_dataset.")
        else:
            print("Using pre-loaded train_dataset.")

    if test_ds is None:
        print(f"Generating test_dataset for experiment: {experiment}...")
        if experiment == "mod":
            test_ds = MetaModuloDataset(
                n_tasks=20 if n_support is not None else 100,
                skip=skip_param,
                train=False,
                data=data_type,
                model=model_arch,
                n_support=n_support,
            )
        elif experiment == "concept":
            test_ds = MetaBitConceptsDataset(
                n_tasks=10 if n_support is not None else 100,
                data=data_type,
                model=model_arch,
                n_support=n_support,
                num_features=num_concept_features,
                pcfg_max_depth=pcfg_max_depth
            )
        elif experiment == "omniglot":
            test_ds = Omniglot(
                n_tasks=10 if n_support is not None else 100,
                model=model_arch,
                train=False,
                alphabet=alphabet # Omniglot test set is fixed, alphabet might be for consistency if train used it
            )
        else:
            raise ValueError(f"Experiment '{experiment}' for test_dataset generation unrecognized.")
        generated_test = True
        print("Finished generating test_dataset.")
    else:
        print("Using pre-loaded test_dataset.")

    if val_ds is None:
        print(f"Generating val_dataset for experiment: {experiment}...")
        if experiment == "mod":
            val_ds = MetaModuloDataset(
                n_tasks=20 if n_support is not None else 100,
                skip=skip_param,
                train=True, # Validation set typically uses train=True split or generation logic
                data=data_type,
                model=model_arch,
                n_support=n_support,
            )
        elif experiment == "concept":
            val_ds = MetaBitConceptsDataset(
                n_tasks=10 if n_support is not None else 100,
                data=data_type,
                model=model_arch,
                n_support=n_support,
                num_features=num_concept_features,
                pcfg_max_depth=pcfg_max_depth
            )
        elif experiment == "omniglot":
            val_ds = Omniglot(
                n_tasks=10 if n_support is not None else 100,
                model=model_arch,
                train=True, # Validation from training distribution
                alphabet=alphabet
            )
        else:
            raise ValueError(f"Experiment '{experiment}' for val_dataset generation unrecognized.")
        generated_val = True
        print("Finished generating val_dataset.")
    else:
        print("Using pre-loaded val_dataset.")

    # Save datasets if paths are provided and they were generated in this call
    if save_train_path and generated_train:
        print(f"Saving train_dataset to {save_train_path}...")
        os.makedirs(os.path.dirname(save_train_path), exist_ok=True)
        torch.save(train_ds, save_train_path)
        print(f"Successfully saved train_dataset to {save_train_path}")
    if save_val_path and generated_val:
        print(f"Saving val_dataset to {save_val_path}...")
        os.makedirs(os.path.dirname(save_val_path), exist_ok=True)
        torch.save(val_ds, save_val_path)
        print(f"Successfully saved val_dataset to {save_val_path}")
    if save_test_path and generated_test:
        print(f"Saving test_dataset to {save_test_path}...")
        os.makedirs(os.path.dirname(save_test_path), exist_ok=True)
        torch.save(test_ds, save_test_path)
        print(f"Successfully saved test_dataset to {save_test_path}")
    
    # Return datasets based on n_support
    if n_support is not None:
        if test_ds is None or val_ds is None:
            # This should ideally be caught by FileNotFoundError if loading was specified,
            # or indicate a bug in generation if generation was expected.
            error_msg = "test_ds or val_ds is None in n_support mode. "
            if load_test_path and test_ds is None: error_msg += f"Failed to load from {load_test_path}. "
            if load_val_path and val_ds is None: error_msg += f"Failed to load from {load_val_path}. "
            if not (load_test_path or load_val_path): error_msg += "Generation might have failed. "
            raise ValueError(error_msg)
        return test_ds, val_ds
    else:
        if train_ds is None or test_ds is None or val_ds is None:
            error_msg = "train_ds, test_ds, or val_ds is None. "
            # Similar detailed error message could be constructed here.
            raise ValueError(error_msg + "Check load paths or generation logic.")
        return train_ds, test_ds, val_ds


def init_model(
    m, data_type, index, verbose: bool = False, channels: int = 1, bits: int = 8, n_output: int = 1,
):
    if data_type == "image":
        n_input = 32 * 32 * channels if m == "mlp" else 16 * channels
    elif data_type == "bits":
        n_input = bits if m == "mlp" else 1
    elif data_type == "number":
        n_input = 1
    else:
        raise ValueError("Data Type unrecognized.")

    if m == "mlp":
        n_hidden, n_layers = MLP_PARAMS[index]
        model = MLP(
            n_input=n_input,
            n_output=n_output,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_input_channels=channels,
        )
    elif m == "cnn":
        n_hidden, n_layers = CNN_PARAMS[index]
        model = CNN(n_input_channels=channels, n_output=n_output, n_hiddens=n_hidden, n_layers=n_layers)
    elif m == "lstm":
        n_hidden, n_layers = LSTM_PARAMS[index]
        model = LSTM(n_input=n_input, n_output=n_output, n_hidden=n_hidden, n_layers=n_layers)
    elif m == "transformer":
        n_hidden, n_layers = TRANSFORMER_PARAMS[index]
        model = Transformer(
            n_input=n_input,
            n_output=n_output,
            d_model=n_hidden,
            dim_feedforward=2 * n_hidden,
            num_layers=n_layers,
        )
    else:
        raise ValueError("Model unrecognized.")

    if verbose:
        print(f"Initialized {m} with {n_hidden} hidden units and {n_layers} layers...")
    return model
