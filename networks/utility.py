import numpy as np
import torch
from torch import nn

def identify_test_place_in_sequence(test_sequence, total_number_of_tests):
    adapted_sequence = test_sequence.copy()
    for i in range(total_number_of_tests):
        if i > len(test_sequence) - 1:
            adapted_sequence.append(-1)
    return adapted_sequence

def observation_to_tensor(obs, action_for_observation=None, total_number_of_tests=None):
    action_tensor = None
    if action_for_observation is not None:
        action_embedding = nn.Embedding(total_number_of_tests, 8)
        action_tensor = action_embedding(torch.tensor(action_for_observation))
    test_index_tensor = torch.tensor([obs.test_index], dtype=torch.float32)
    test_sequence_precise_tensor = torch.tensor(identify_test_place_in_sequence(obs.test_sequence, total_number_of_tests), dtype=torch.float32)
    mutant_operator_tensor = torch.tensor([obs.mutant_operator], dtype=torch.float32)
    num_of_mutants_same_operator_killed_tensor = torch.tensor([obs.num_of_mutants_same_operator_killed], dtype=torch.float32)
    if action_for_observation is None:
        return torch.cat((test_index_tensor, test_sequence_precise_tensor, mutant_operator_tensor, num_of_mutants_same_operator_killed_tensor))
    return torch.cat((action_tensor, test_sequence_precise_tensor, mutant_operator_tensor)) #used in the observation network to predict if the mutant is killed or not


def inference(state_tensor, model):
    tensor_in = state_tensor
    tensor_in = tensor_in[None]  # Wrap one outer dimension (as for a batch)
    with torch.no_grad():
        output = model(tensor_in.to('cuda' if torch.cuda.is_available() else 'cpu'))
    output = output.squeeze()  # Remove the outermost batch-size dimension
    return output.item() if model.__class__.__name__ == 'ValueNN' or model.__class__.__name__ == 'ObservationNN' \
        else torch.softmax(output, dim=-1)


def training_model(model, inputs, targets, opt, loss_function, scheduler=None):

    if model.__class__.__name__ == 'ValueNN':
        print("Training ValueNN model...")
    elif model.__class__.__name__ == 'PolicyNN':
        print("Training PolicyNN model...")
    else:
        print("Training ObservationNN model...")

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