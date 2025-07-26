from bayesflow.trainers import Trainer
from bayesflow.helper_functions import loss_to_string, backprop_step
import logging
from tqdm import tqdm
import tensorflow as tf

from bayesflow.helper_classes import (
    EarlyStopper,
    LossHistory,
    MemoryReplayBuffer,
    MultiSimulationDataset,
    SimulationDataset,
    SimulationMemory,
)

class CustomTrainer(Trainer):
    def train_offline(
        self,
        simulations_dict=None,
        train_dataset=None,
        epochs=1,
        optimizer=None,
        batch_size=64,
        save_checkpoint=True,
        reuse_optimizer=False,
        early_stopping=False,
        validation_dataset=None,
        validation_sims=None,
        val_freq=5,
        use_autograph=True,
        **kwargs
    ):
        if simulations_dict is not None and train_dataset is not None:
            raise ValueError("Pass only one of simulations_dict or train_dataset.")
        elif simulations_dict is not None:
            data_set = SimulationDataset(simulations_dict, batch_size)
            use_tf_dataset = False
            print("[INFO] Using simulations_dict with SimulationDataset.")
        elif train_dataset is not None:
            data_set = train_dataset
            use_tf_dataset = True
            print("[INFO] Using tf.data.Dataset for training.")
        else:
            raise ValueError("Either simulations_dict or train_dataset must be provided.")
    
        # Compile update function, if specified
        if use_autograph:
              _backprop_step = tf.function(backprop_step, reduce_retracing=True)
        else:
               _backprop_step = backprop_step
          
        self._setup_optimizer(optimizer, epochs, 5)
        self.loss_history.start_new_run()
    
        for ep in range(1, epochs + 1):
          with tqdm(
                    total=tf.data.experimental.cardinality(data_set).numpy(), desc="Training epoch {}".format(ep), mininterval=TQDM_MININTERVAL
          ) as p_bar:
            for bi, forward_dict in enumerate(data_set, start=1):
                if not use_tf_dataset:
                    input_dict = self.configurator(forward_dict, **kwargs.pop("conf_args", {}))
                else:
                    input_dict = self.configurator(forward_dict)
    
                loss = self._train_step(batch_size, _backprop_step, input_dict, **kwargs)
                self.loss_history.add_entry(ep, loss)
    
                # Compute running loss
                avg_dict = self.loss_history.get_running_losses(ep)
    
                # Extract current learning rate
                lr = extract_current_lr(self.optimizer)
    
                # Format for display on progress bar
                disp_str = format_loss_string(ep, bi, loss, avg_dict, lr=lr, it_str="Batch")
    
                # Update progress
                p_bar.set_postfix_str(disp_str, refresh=False)
                p_bar.update(1)
    
    
          # Store and compute validation loss, if specified
          self._save_trainer(save_checkpoint)
          if validation_sims is not None or validation_dataset is not None and ep / val_freq == 0:
              self._validation(
                  ep,
                  validation_sims=validation_sims,
                  validation_dataset=validation_dataset,
                  **kwargs,
              )
    
          # Check early stopping, if specified
          if self._check_early_stopping(early_stopper):
                break
              
        # Remove optimizer reference, if not set as persistent
        if not reuse_optimizer:
            self.optimizer = None
        return self.loss_history.get_plottable()
    
    
    
    def _validation(self, ep, validation_sims=None, validation_dataset=None, **kwargs):
        """Helper method to compute validation loss either from dict or dataset."""
        if validation_sims is not None and validation_dataset is not None:
            raise ValueError("Provide only one of validation_sims or validation_dataset.")
    
        val_loss = None
    
        if validation_sims is not None:
            conf = self.configurator(validation_sims, **kwargs.pop("val_conf_args", {}))
            val_loss = self.amortizer.compute_loss(conf, **kwargs.pop("net_args", {}))
    
        elif validation_dataset is not None:
            val_losses = []
            for val_batch in validation_dataset:
                input_dict = self.configurator(val_batch)
                batch_loss = self.amortizer.compute_loss(input_dict)
                val_losses.append(batch_loss)
    
            # Aggregate losses over batches
            if isinstance(val_losses[0], dict):
                val_loss = {
                    k: tf.reduce_mean(tf.stack([l[k] for l in val_losses]))
                    for k in val_losses[0]
                }
            else:
                val_loss = tf.reduce_mean(tf.stack(val_losses))

        if val_loss is not None:
            self.loss_history.add_val_entry(ep, val_loss)
            val_loss_str = loss_to_string(ep, val_loss)
            logging.getLogger().info(val_loss_str)
