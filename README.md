# MLClock

An experiment to train a model to read the current time off of analog clocks.

![demo](https://github.com/KittyMac/MLClock/blob/master/meta/demo.jpg?raw=true)


How it works
------------

### model.py

this code is responsible for generating the model architecture for use in training. the current model works by accepting a 128x128 image as input, and 72 or 132 outputs. The outputs are a multi-hot array corresponding to the hour, minute, and/or second that the clock face represents. So if it is 4:46 as in the above photo, we would have the 3 index and the 58 index be hot. Not all analog clocks include a seconds hand, so you can tweak that by setting INCLUDE_SECONDS_HAND.


### data.py

this code is responsible for generating augmented training data. It does this by using images taken of an existing clock, and composing them such that a clock image can be generated with near realistic accuracy. data.py also includes some data augemntation by adjusting coloration, rotation, and offset of the generated clock face.  

if you run data.py directly it will generate some sample images to /tmp for manual verification.

![demo2](https://github.com/KittyMac/MLClock/blob/master/meta/demo2.png?raw=true)


### train.py

for training the model, simply run `python train.py learn` and 1m iterations of randomly generate clock faces will be trained.  once you have some weights trained, you can run `python train.py convert` to export a new coreml model, `python train.py test` to run an comprehensive accuracy test, or `python train.py test2 4:15:25` with a time to test against a specific time.


TODO
------------

1. experiment to find the best model architecture
2. add object localization to the app so we can smartly crop the clock face
3. add more image augmentation to training to allow more off angled shots of the face