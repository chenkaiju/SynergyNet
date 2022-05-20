1. tfds build

cd aflw2000_tfds

tfds build --data_dir="./"

2. tfds load

import aflw2000_tfds

ds = tfds.load('aflw2000_tfds', data_dir='aflw2000_tfds', as_supervised=True) #as_supervised=False if you need keys
