"""custom_dataset dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import pickle
import numpy as np
import os


# TODO(custom_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(custom_dataset): BibTeX citation
_CITATION = """
"""


class CustomDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for custom_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(custom_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(120, 120, 3)),
            'param': tfds.features.Tensor(shape=(62,), dtype=tf.float64),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'param'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(custom_dataset): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    
    root = "../" # Synergy folder
    
    images_fileDir = os.path.join(root, "train_aug_120x120")
    images_filelistpath = os.path.join(root, "3dmm_data", "train_aug_120x120.list.train")
    params_filepath = os.path.join(root, "3dmm_data", "param_all_norm_v201.pkl")

    lines = Path(images_filelistpath).read_text().strip().split('\n')
    img_paths = [os.path.join(images_fileDir, s) for s in lines]
    params = self._load_param(params_filepath)[:,:62] #12 pose, 40 shape, 10 expression, 40 texture
        
    # TODO(custom_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(img_paths, params),
    }

  def _generate_examples(self, paths, params):
    """Yields examples."""
    # TODO(custom_dataset): Yields (key, example) tuples from the dataset
    for i in range(len(paths)):
      path = paths[i]
      param = params[i]
      yield i, {
          'image': path,
          'param': param,
      }

  def _load_param(self, fp):
      suffix = self._get_suffix(fp)
      if suffix == 'npy':
          return np.load(fp)
      elif suffix == 'pkl':
          return pickle.load(open(fp, 'rb'))
        
  def _get_suffix(self, filename):
      pos = filename.rfind('.')
      if pos == -1:
          return ''
      return filename[pos + 1:]
      

if __name__ == "__main__":
  
  ds = CustomDataset()
  dl =tfds.download.DownloadManager
  ds._split_generators(dl)
