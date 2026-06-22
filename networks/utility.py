import numpy as np
import torch
import logging
from torch import nn

def identify_test_place_in_sequence(test_sequence, total_number_of_tests):
    adapted_sequence = test_sequence.copy()
    for i in range(total_number_of_tests):
        if i > len(test_sequence) - 1:
            adapted_sequence.append(-1)
    return adapted_sequence

def observation_to_tensor(obs, total_number_of_tests=None):
    test_index_tensor = torch.tensor([obs.test_index], dtype=torch.float32)
    test_sequence_precise_tensor = torch.tensor(identify_test_place_in_sequence(obs.test_sequence, total_number_of_tests), dtype=torch.float32)
    mutant_operator_tensor = torch.tensor([obs.mutant_operator], dtype=torch.float32)
    return torch.cat((test_index_tensor, test_sequence_precise_tensor, mutant_operator_tensor))


def inference(state_tensor, model):
    tensor_in = state_tensor
    tensor_in = tensor_in[None]  # Wrap one outer dimension (as for a batch)
    with torch.no_grad():
        output = model(tensor_in.to('cuda' if torch.cuda.is_available() else 'cpu'))
    output = output.squeeze()  # Remove the outermost batch-size dimension
    return output.item() if model.__class__.__name__ == 'ValueNN' else torch.softmax(output, dim=-1)


def training_model(model, inputs, targets, opt, loss_function, scheduler=None):

    if model.__class__.__name__ == 'ValueNN':
        logging.debug("Training ValueNN model...")
    elif model.__class__.__name__ == 'PolicyNN':
        logging.debug("Training PolicyNN model...")

    model.train()
    losses = []
    for batch, (X, y) in enumerate(zip(inputs, targets)):
        opt.zero_grad()  # resetting the gradients
        pred = model(X.to('cuda' if torch.cuda.is_available() else 'cpu'))  # forward pass

        loss = loss_function(pred, y.to('cuda' if torch.cuda.is_available() else 'cpu'))
        losses.append(loss.item())
        loss.backward()  # computing the gradient
        opt.step()  # changing the weights of the network

        if scheduler is not None:
            scheduler.step()

    return np.mean(losses)