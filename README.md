# CNN-Fixations-PyTorch
An impementation of CNN Fixations for PyTorch with automatic network structure detection.

## Run instructions:
For running the mnist example use command:
`python .\mnist.py --test-batch-size [batch_size] --load-model "mnist_cnn.pt" --no-cuda`

## TODO:
* Make the code run faster/more efficiently
* Add support for converting a super module into its sub modules when running fixations.
* Fix fixations for (2+1)D convolutions
* Add support for more basic PyTorch modules such as:
  * Dropout
  * Various kind of pooling
* Add the ability to register custom fixation functions linked to which module it should execute on. This would allow users to have working fixations for custom PyTorch modules.