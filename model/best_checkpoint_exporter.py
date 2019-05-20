"""
Adapted from https://github.com/bluecamel/best_checkpoint_copier
Copyright (c) 2018 Branton Davis
"""

import glob
import os
import shutil
import tensorflow as tf


class Checkpoint(object):
    dir = None
    file = None
    score = None
    path = None

    def __init__(self, path, score):
        self.dir = os.path.dirname(path)
        self.file = os.path.basename(path)
        self.score = score
        self.path = path


class BestCheckpointExporter(tf.estimator.Exporter):
    """This class keeps the checkpoints of the best runs."""
    checkpoints = None
    checkpoints_to_keep = None
    compare_fn = None
    name = None
    score_metric = None
    sort_key_fn = None
    sort_reverse = None

    def __init__(self,
                 name="best_checkpoints",
                 checkpoints_to_keep=5,
                 score_metric='Loss/total_loss',
                 compare_fn=lambda x,y: x.score < y.score,
                 sort_key_fn=lambda x: x.score,
                 sort_reverse=False):
        """Create the exporter"""
        self.checkpoints = []
        self.checkpoints_to_keep = checkpoints_to_keep
        self.compare_fn = compare_fn
        self.name = name
        self.score_metric = score_metric
        self.sort_key_fn = sort_key_fn
        self.sort_reverse = sort_reverse
        super(BestCheckpointExporter, self).__init__()

    def _log(self, statement):
        tf.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

    def export(self, estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
        self._log('Export checkpoint {}'.format(checkpoint_path))
        score = float(eval_result[self.score_metric])
        checkpoint = Checkpoint(path=checkpoint_path, score=score)

        if len(self.checkpoints) < self.checkpoints_to_keep \
            or self.compare_fn(checkpoint, self.checkpoints[-1]):
            # Keep the checkpoint
            self._log('Keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))
            self.checkpoints.append(checkpoint)
            self.checkpoints = sorted(
                self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)
            # The destination directory (make if necessary)
            destination_dir = os.path.join(checkpoint.dir, self.name)
            os.makedirs(destination_dir, exist_ok=True)
            # Copy the checkpoint
            for file in glob.glob(r'{}*'.format(checkpoint.path)):
                self._log('Copying {} to {}'.format(file, destination_dir))
                shutil.copy(file, destination_dir)
            # Prune checkpoints
            for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
                self._log('Removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))
                old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
                for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
                    os.remove(file)
            self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]
        else:
            # Skip the checkpoint
            self._log('Skipping checkpoint {}'.format(checkpoint.path))
