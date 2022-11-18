 
## Notice
- model files such as m1.h5 files are not included due to storage capacity issue. you should create h5 file locally
- model consists of 3 layers including one hidden layer. Number of layers and nodes may be modified

## Getting Started
- download code
- insert data files you prepared. Initial setting is 'speech.txt' and 'non-speech.txt'
- give appropriate parameters according to your input file. If your input shape is (N, 4872) parameter A and B should be...
- run and get model saved in h5 fomat

## Conversion for javascript usage
You should have your system ready for conversion. specific tensorflow version and other stuffs needed. check google for detailed information
```bash
tensorflowjs_converter --input_format keras ./m2.h5 ./tfjs_files
```
will create a json file and bin file(s) into tfjs_files directory
