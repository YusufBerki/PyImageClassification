# Callbacks

A callback is an object that can perform actions at various stages of training (e.g. at the start or end of an epoch, before or after a single batch, etc).

You can use callbacks to:
- Write TensorBoard logs after every batch of training to monitor your metrics
- Periodically save your model to disk
- Do early stopping
- Reduce learning rate when a metric has stopped improving.

Descriptions in this document are taken from [Keras official website](https://keras.io/). You can check [this](https://keras.io/api/callbacks/) page for more information about callback functions.

## TensorBoard
[TensorBoard](https://www.tensorflow.org/tensorboard) is a visualization tool provided with [TensorFlow](https://www.tensorflow.org/).

This callback logs events for TensorBoard, including:

-   Metrics summary plots
-   Training graph visualization
-   Activation histograms
-   Sampled profiling

### Usage
Basically TensorBoard can enable like:
> python train.py --tensorboard True

Used in this way, tensorflow logs are saved in the *"./tensorboard_logs"* directory.

To open the TensorBoard, run:
> tensorboard --logdir tensorboard_logs

This should print that TensorBoard has started. Next, connect to  [http://localhost:6006](http://localhost:6006/).

### Arguments

-   **tensorboard_log_dir**: the path of the directory where to save the log files to be parsed by TensorBoard. This directory should not be reused by any other callbacks.
-   **tensorboard_histogram_freq**: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
-   **tensorboard_write_graph**: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
-   **tensorboard_write_images**: whether to write model weights to visualize as image in TensorBoard.
-   **tensorboard_write_steps_per_second**: whether to log the training steps per second into Tensorboard. This supports both epoch and batch frequency logging.
-   **tensorboard_update_freq**:  batch  or  epoch  or integer. When using  batch, writes the losses and metrics to TensorBoard after each batch. The same applies for  epoch. If using an integer, let's say  1000, the callback will write the metrics and losses to TensorBoard every 1000 batches. Note that writing too frequently to TensorBoard can slow down your training.
-   **tensorboard_profile_batch**: Profile the batch(es) to sample compute characteristics. profile_batch must be a non-negative integer or a tuple of integers. A pair of positive integers signify a range of batches to profile. By default, it will profile the second batch. Set profile_batch=0 to disable profiling.
-   **tensorboard_embeddings_freq**: frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings won't be visualized.

You can find more information about TensorBoard  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard) and [here](https://keras.io/api/callbacks/tensorboard/).
## ReduceLROnPlateau
Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

### Usage

> python train.py --reduce_lr True

### Arguments

-   **reduce_lr_monitor**: quantity to be monitored.
-   **reduce_lr_factor**: factor by which the learning rate will be reduced.  `new_lr = lr * factor`.
-   **reduce_lr_patience**: number of epochs with no improvement after which learning rate will be reduced.
-   **reduce_lr_verbose**: int. 0: quiet, 1: update messages.
-   **reduce_lr_mode**: one of  `{'auto', 'min', 'max'}`. In  `'min'`  mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in  `'max'`  mode it will be reduced when the quantity monitored has stopped increasing; in  `'auto'`  mode, the direction is automatically inferred from the name of the monitored quantity.
-   **reduce_lr_min_delta**: threshold for measuring the new optimum, to only focus on significant changes.
-   **reduce_lr_cooldown**: number of epochs to wait before resuming normal operation after lr has been reduced.
-   **reduce_lr_min_lr**: lower bound on the learning rate.

## ModelCheckpoint
Callback to save the Keras model or model weights at some frequency.

`ModelCheckpoint`  callback is used in conjunction with training to save a model or weights (in a checkpoint file) at some interval, so the model or weights can be loaded later to continue the training from the state saved.

A few options this callback provides include:

-   Whether to only keep the model that has achieved the "best performance" so far, or whether to save the model at the end of every epoch regardless of performance.
-   Definition of 'best'; which quantity to monitor and whether it should be maximized or minimized.
-   The frequency it should save at. Currently, the callback supports saving at the end of every epoch, or after a fixed number of training batches.
-   Whether only weights are saved, or the whole model is saved.

Note: If you get  `WARNING:tensorflow:Can save best model only with <name> available, skipping`  see the description of the  `monitor`  argument for details on how to get this right.

### Usage

> python train.py --model_checkpoint True

### Arguments

-   **model_checkpoint_filepath**: path to save the model file. filepath can contain named formatting options, which will be filled the value of  `epoch`  and keys in  `logs`  (passed in  `on_epoch_end`). For example: if  `filepath`  is  `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model checkpoints will be saved with the epoch number and the validation loss in the filename. The directory of the filepath should not be reused by any other callbacks to avoid conflicts.
-   **model_checkpoint_monitor**: The metric name to monitor. Typically the metrics are set by the  `Model.compile`  method. Note:
    
    -   Prefix the name with  `"val_`" to monitor validation metrics.
    -   Use  `"loss"`  or "`val_loss`" to monitor the model's total loss.
    -   If you specify metrics as strings, like  `"accuracy"`, pass the same string (with or without the  `"val_"`  prefix).
   
   -   **model_checkpoint_verbose**: verbosity mode, 0 or 1.
   -   **model_checkpoint_save_best_only**: if  `save_best_only=True`, it only saves when the model is considered the "best" and the latest best model according to the quantity monitored will not be overwritten. If  `filepath`  doesn't contain formatting options like  `{epoch}`  then  `filepath`  will be overwritten by each new better model.
   -   **model_checkpoint_mode**: one of `{'auto', 'min', 'max'}`. If  `save_best_only=True`, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For  `val_acc`, this should be  `max`, for  `val_loss`  this should be  `min`, etc. In  `auto`  mode, the mode is set to  `max`  if the quantities monitored are 'acc' or start with 'fmeasure' and are set to  `min`  for the rest of the quantities.
   -   **model_checkpoint_save_weights_only**: if True, then only the model's weights will be saved, else the full model is saved.
   -   **model_checkpoint_save_freq**:  `'epoch'`  or integer. When using  `'epoch'`, the callback saves the model after each epoch. When using integer, the callback saves the model at end of this many batches.
## EarlyStopping
Stop training when a monitored metric has stopped improving.

Assuming the goal of a training is to minimize the loss. With this, the metric to be monitored would be  `'loss'`, and mode would be  `'min'`. A training loop will check at end of every epoch whether the loss is no longer decreasing, considering the  `min_delta`  and  `patience`  if applicable. Once it's found no longer decreasing, the training terminates.

The quantity to be monitored needs to be available in  `logs`  dict. To make it so, pass the loss or metrics.
### Usage

> python train.py --early_stopping True

### Arguments

-   **early_stopping_monitor**: Quantity to be monitored.
-   **early_stopping_min_delta**: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
-   **early_stopping_patience**: Number of epochs with no improvement after which training will be stopped.
-   **early_stopping_verbose**: verbosity mode.
-   **early_stopping_mode**: One of  `{"auto", "min", "max"}`. In  `min`  mode, training will stop when the quantity monitored has stopped decreasing; in  `"max"`  mode it will stop when the quantity monitored has stopped increasing; in  `"auto"`  mode, the direction is automatically inferred from the name of the monitored quantity.
-   **early_stopping_baseline**: Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
-   **early_stopping_restore_best_weights**: Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used. An epoch will be restored regardless of the performance relative to the  `baseline`. If no epoch improves on  `baseline`, training will run for  `patience`  epochs and restore weights from the best epoch in that set.

# Todo
-   [ ] LearningRateScheduler [:link:](https://keras.io/api/callbacks/learning_rate_scheduler)
-   [ ] RemoteMonitor [:link:](https://keras.io/api/callbacks/remote_monitor)
-   [ ] LambdaCallback [:link:](https://keras.io/api/callbacks/lambda_callback)
-   [ ] TerminateOnNaN [:link:](https://keras.io/api/callbacks/terminate_on_nan)
-   [ ] CSVLogger [:link:](https://keras.io/api/callbacks/csv_logger)
-   [ ] ProgbarLogger [:link:](https://keras.io/api/callbacks/progbar_logger)
-   [x] ~~ModelCheckpoint~~ [:link:](https://keras.io/api/callbacks/model_checkpoint)
-   [x] ~~TensorBoard~~ [:link:](https://keras.io/api/callbacks/tensorboard)
-   [x] ~~EarlyStopping~~ [:link:](https://keras.io/api/callbacks/early_stopping)
-   [x] ~~ReduceLROnPlateau~~ [:link:](https://keras.io/api/callbacks/reduce_lr_on_plateau)