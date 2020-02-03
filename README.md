# deep-intake-detection: Automatic detection of intake gestures from video

Implementation of various deep neural networks for detection of food and drink intake gestures from video.
We use recordings from a 360-degree camera to predict frame-level labels.
Find our published paper [here](https://ieeexplore.ieee.org/document/8853283).

## Prepare data for TensorFlow

We use `.tfrecord` files to feed data into the TensorFlow model.
This includes [TV-L<sup>1</sup> optical flow](https://pequan.lip6.fr/~bereziat/cours/master/vision/papers/zach07.pdf), which is stored in the file alongside with raw frames and labels.
A script and instructions for generating these files can be found at the [video-sensor-processing repository](https://github.com/prouast/video-sensor-processing).

## Train and evaluate deep neural network classifier

Train and evaluate the TensorFlow classifier using the `.tfrecord` files in train/eval folders.

```
$ python oreba_main.py
```

The best checkpoints based on the evaluation set are automatically saved using [best_checkpoint_copier](https://github.com/bluecamel/best_checkpoint_copier).

Specify model architecture and other settings using the flags introduced below, see the [code](https://github.com/prouast/deep-intake-detection/blob/master/model/oreba_main.py#L21) for more details.

### What are the flags?

| Argument | Description |
| --- | --- |
| --base_learning_rate | Choose the default learning rate for Adam optimizer |
| --batch_size | Batch size used for training. |
| --data_format | Set the data format used in the model (defaults to channels_last). |
| --dtype | TensorFlow datatype used for calculations {fp16, fp32} (defaults to fp32). |
| --eval_dir | Directory for eval data (defaults to 'eval') |
| --finetune_only | If using warmstart, specify what types of layer should be finetuned (defaults to all) |
| --label_category | Label category for classification task (defaults to 1) |
| --mode | Which mode to start in {train_and_evaluate, predict_and_export_csv, predict_and_export_tfrecord} (defaults to train_and_evaluate) |
| --model | Select the model: {oreba_2d_cnn, oreba_cnn_lstm, oreba_two_stream, oreba_slowfast, resnet_2d_cnn, resnet_3d_cnn, resnet_cnn_lstm, resnet_two_stream, resnet_slowfast} |
| --model_dir | Output directory for model and training stats (defaults to 'run') |
| --num_frames | Specify how many frames are in train data |
| --num_parallel_calls | Number of parallel calls in input pipeline (defaults to None) |
| --num_sequences | Specify how many sequences are in train data |
| --save_checkpoints_steps | Save checkpoints every x steps (defaults to 100) |
| --save_summary_steps | Save summaries every x steps (defaults to 10) |
| --sequence_shift | Shift taken in sequence generation (defaults to 1) |
| --shuffle_buffer | Buffer size for shuffling (defaults to 25000) |
| --slowfast_alpha | Alpha parameter in SlowFast (defaults to 4) |
| --slowfast_beta | Beta parameter in SlowFast (defaults to 0.25) |
| --train_dir | Directory for training data (defaults to 'train') |
| --train_epochs | Number of training epochs (defaults to 40) |
| --use_distribution | Use tf.distribute.MirroredStrategy (defaults to False) |
| --use_flows | Use optical flow as input (defaults to False) |
| --use_frames | Use frames as input (defaults to True) |
| --use_sequence_input | Use sequences as input instead of single examples (defaults to False) |
| --use_sequence_loss | Use sequence-to-sequence loss instead of sequence-to-one loss (defaults to False) |
| --warmstart_dir | Directory with checkpoints for warmstart (defaults to None) |

### Which models can be warmstarted and how?

- `resnet_2d_cnn` can be warmstarted with `resnet` (ResNet-50 v2 checkpoints available [here](https://github.com/tensorflow/models/tree/master/official/resnet))
- `oreba_cnn_lstm` can be warmstarted with `oreba_2d_cnn`
- `resnet_cnn_lstm` can be warmstarted with `resnet_2d_cnn`
- `oreba_two_stream` can be warmstarted with `oreba_2d_cnn`
- `resnet_two_stream` can be warmstarted with `resnet_2d_cnn`

Checkpoints for these models have to be present in the directory specificied in `--warmstart_dir`.

### Evaluate F1 score

To evaluate F1 score on exported frame-level probabilities in `folder`, run

```
$ python eval.py --prob_dir=folder
```

You can use the flag `--threshold` to specify the threshold above which a frame will be classified as Intake, or use `--mode=estimate` to let the model search for the threshold between `--min_threshold` and `--max_threshold` that maximizes F1.

## Results on OREBA-LBY dataset

The following models have been trained on the training set of 62 participants to distinguish between idle and intake frames.
Unweighted average recall (UAR) is the average classification accuracy for idle and intake frames on the test set of 20 participants.
F1 is based on actual detection of individual intake gestures on the test set based on the evaluation scheme proposed by [Kyritsis et al.](https://ieeexplore.ieee.org/abstract/document/8606156).   

| Model | Features | UAR | F1 |
| --- | ---  | --- | --- |
| Small 2D CNN | frames | 82.63% | 0.674 |
| Small 2D CNN | flows | 71.76% | 0.478 |
| ResNet-50 2D CNN | frames | 86.39% | 0.795 |
| ResNet-50 2D CNN | flows | 71.34% | 0.461 |
| Small 3D CNN | frames | 87.54% | 0.798 |
| Small CNN-LSTM | frames | 83.36% | 0.755 |
| Small Two-Stream | frames and flows | 81.96% | 0.700 |
| Small SlowFast | frames | 88.71% | 0.803 |
| ResNet-50 3D CNN | frames | 88.77% | 0.840 |
| ResNet-50 CNN-LSTM | frames | 89.74% | 0.856 |
| ResNet-50 Two-Stream | frames and flows | 85.25% | 0.836 |
| ResNet-50 SlowFast | frames | 89.01% | 0.858 |
