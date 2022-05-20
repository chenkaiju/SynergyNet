1. tfds build

cd tddfa_tfds

tfds build --data_dir="./"

2. tfds load

import tddfa_tfds

ds = tfds.load('tddfa_tfds', data_dir='tddfa_tfds', as_supervised=True) #as_supervised=False if you need keys
