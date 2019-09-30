'''
CNN Fixations for Pytorch

To run CNN Fixations on a trained mnist model use command:
python visualize_mnist.py --test-batch-size [batch_size] --visualize-model "models/mnist_cnn.pt" --no-cuda
'''
import torch

from models import mnist
from cnn_fixations.fixations import Fixations, to_cpu
import cnn_fixations.utils as U


def compute_fixations(predictions, model, fx, device):
    predictions = to_cpu(U.zip_batch_number(predictions))
    model = to_cpu(model)

    # Set up fixation functions for each layer
    points = fx.fully_connected(predictions, model.fc2)
    points = fx.fully_connected(points, model.fc1)
    points = fx.maxpool(points, model.max_pool)
    points = fx.convolution(points, model.conv2)
    points = fx.maxpool(points, model.max_pool)
    points = fx.convolution(points, model.conv1)

    model = model.to(device)
    points = U.chunk_batch(points)  # Groups points by batch
    return points

if __name__ == '__main__':
    args, model, _, _, test_loader, device = mnist.main()
    path = args.visualize_model
    if path is not None:
        model.load_state_dict(torch.load(path))
        fx = Fixations()  # Can be reused, thus no model argument

        # Run inference on a trained model and record activations
        with fx.record_activations(model):
            inputs, predictions, targets = mnist.test(args, model, device, test_loader, visualize=True)

        # Computer CNN fixations and show them
        points = compute_fixations(predictions, model, fx, device)
        for i in range(len(points)):  # Iterate over batch
            U.visualize(inputs[i], points[i], diag_percent=0.1, image_label=targets[i], prediction=predictions[i], sigma=2)
