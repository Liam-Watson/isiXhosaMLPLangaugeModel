# ixiXhosa Feed Forward Neural Network Language Model
Github repo: https://github.com/Liam-Watson/isiXhosaMLPLangaugeModel
## Enviroment instructions 
This project is implimented in python3.7.13 on Ubuntu 22.04 using an anaconda enviroment. The required packages and versions listed by enviroment are as listed:
* pytorch                   1.11.0          py3.7_cuda11.3_cudnn8.2.0_0    pytorch
* matplotlib                3.5.2            py37h89c1867_0    conda-forge
* numpy                     1.21.6                   pypi_0    pypi
* numpy-base                1.21.5           py37ha15fc14_3    anaconda
Note: These will require many dependences which are exluded here. If additional enviroment information is needed please contact. 
The pytorch version will vary depending on GPU/CPU requirements - I have an RTX 1660ti mobile. (cuda11.3)

## Project structure
Included in the project are the development python files, training/validation/test data, output text files, models.
* model.py contains the feed forward neural network model
* preprocessing.py contains the methods needed for loading and processing train/valid/test data.
* train.py contains the overall training, validation and testing framework with output and commented optimization techniques
* test.py contains code for testing a saved model

## Running the project
* Ensure all required packages are installed
* Training:
	* `python3 train.py`
* Testing:
	* `python3 test.py <MODEL PATH/>`

