# MLClock

An experiment to train a model to read the current time off of analog clocks.

![demo](https://github.com/KittyMac/MLClock/blob/master/meta/demo.gif?raw=true)


## Data Generation

### data.py

this code is responsible for generating augmented training data. It does double duty in generating data for clock localzation and also for time detection. It does this by using images taken of an existing clock and composing them such that a clock image can be generated with near realistic accuracy. data.py also includes some data augemntation by adjusting coloration, rotation, and offset of the generated clock face.  

if you run data.py directly it will generate some sample images to /tmp for manual verification.

Sample generated for Time Detection:  
![demo2](https://github.com/KittyMac/MLClock/blob/master/meta/demo2.png?raw=true)

Sample generated for Clock Localization:  
![demo2](https://github.com/KittyMac/MLClock/blob/master/meta/demo3.png?raw=true)


## Time Detection

### model.py

this code is responsible for generating the model architecture for use in training. the current model works by accepting an image as input, and 72 or 132 outputs. The outputs are a multi-hot array corresponding to the hour, minute, and/or second that the clock face represents. So if it is 4:46 as in the above photo, we would have the 3 index and the 58 index be hot. Not all analog clocks include a seconds hand, so you can tweak that by setting INCLUDE_SECONDS_HAND.


### train.py

for training the model, simply run `python train.py learn` and 1m iterations of randomly generate clock faces will be trained.  once you have some weights trained, you can run `python train.py convert` to export a new coreml model, `python train.py test` to run an comprehensive accuracy test, or `python train.py test2 4:15:25` with a time to test against a specific time.


## Clock Localization

### model.py

this model is used for clock localization (ie finding a close cropping rectangle around clock faces in the image).  The outputs of the model are a multi-hot array which represents the x and y axis of the image broken up into grids (in the current case, the image is 100x100 and the arrays are 100x100). Each grid is set to 1 if the clock is inside that row and column.


### train.py

for training the model, simply run `python train.py learn` and 1m iterations of randomly generated localized clocks.  once you have some weights trained, you can run `python train.py convert` to export a new coreml model for use in the app.





## License

MLClock is free software distributed under the terms of the MIT license, reproduced below. MLClock may be used for any purpose, including commercial purposes, at absolutely no cost. No paperwork, no royalties, no GNU-like "copyleft" restrictions. Just download and enjoy.

Copyright (c) 2018 Rocco Bowling

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.