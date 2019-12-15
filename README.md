# Weather Predicition

4 classes of weather have been predicted with using CNN from the scratch.

  - HAZE
  - SUNNY
  - SNOWY
  - RAINY

# To Be Done

  - Apply Transfer Learning

### Usage

This program is built using the following versions:
Python 3.7
Tensorflow 2.0.0
Keras 2.3.1

Install the dependencies.

```sh
$ pip install keras
$ pip install tensorflow
$ pip install numpy,pandas,pickle
$ pip install opencv-python
$ pip install imutils
```

For training the model

```sh
$ python train.py --dataset dataset --model weather.model --labelbin lb.pickle
```

For testing the blind dataset

```sh
$ python blindtest.py --model weather.model --labelbin lb.pickle --image examples/test
```

For testing only one photo while seeing the results in OpenCV

```sh
$ python classify.py --model weather.model --labelbin lb.pickle --image examples/HAZE_1.png
```

License
----

MIT
