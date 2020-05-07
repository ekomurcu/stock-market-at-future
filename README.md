# Uppsala University Term Project - Stock Market Prediction 

### How to install project: 
 
- Install anaconda 
- Create virtual env 
- Install dependencies with below command. 

```
conda env update --prefix ./env --file environment.yml  --prune
```

- Create `data` directory in project directory 
- Download `.csv` files from there [link](https://www.kaggle.com/ehallmar/daily-historical-stock-prices-1970-2018)
- Add files to `data` directory.
- Go to `load_data.py` and change the `FILE_PATH` variable with your path. 


Predicting the prices of stock market

In this project, the plan is to do stock value prediction. 

It is planned to use the data set whose link is given below. However, the used dataset may be varied according to the will. 
https://www.kaggle.com/ehallmar/daily-historical-stock-prices-1970-2018

In addition to that, this data can be related to Coronavirus, if the time permits. 
For test data, the plan is to train the model with year 1970-2015 betweens such that the accuracy can be tested with a dataset between 2015 and 2018. 

The stock price prediction is done via the advanced regression using Long Term Short Memory (LSTM) and Recurrent Neural Networks (RNN) in Tensoflow.

## Project Specs 

Phase one: Explore Data  

- [x] Loading Data
- [x] Data Exploration 
- [x] Preprocessing 
- [ ] Data Visualisation 

Phase two: Machine Learning

- Comming soon

Phase three: Optimization

- Comming soon

Phase four: Review

- Comming soon

