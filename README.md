# Predicting Median House Values

## Project Overview
This project aims to predict the median value of owner-occupied homes in Boston using a dataset from the U.S. Census Service. The goal is to showcase a multiple linear regression model based on the `rm` (average number of rooms) and `lstat` (percentage of lower status population) variables.

## Dataset
The dataset consists of 506 entries with the following variables:
- `crim`: per capita crime rate by town
- `zn`: proportion of residential land zoned for lots over 25,000 sq. ft
- `indus`: proportion of non-retail business acres per town
- `chas`: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- `nox`: nitric oxide concentration (parts per 10 million)
- `rm`: average number of rooms per dwelling
- `age`: proportion of owner-occupied units built prior to 1940
- `dis`: weighted distances to five Boston employment centers
- `rad`: index of accessibility to radial highways
- `tax`: full-value property tax rate per $10,000
- `ptratio`: pupil-teacher ratio by town
- `b`: 1000(bk — 0.63)², where bk is the proportion of people of African American descent by town
- `lstat`: percentage of lower status of the population
- `medv`: median value of owner-occupied homes in $1000s

### Reference
Harrison, David, and Daniel L. Rubinfeld, "Hedonic Housing Prices and the Demand for Clean Air," Journal of Environmental Economics and Management, Volume 5, (1978), 81-102.

## Objective
Analyze the relationship between variables and build a regression model to predict median home values (`medv`) using `rm` and `lstat`.

## Workflow
1. **Import Libraries**: pandas, numpy, matplotlib, seaborn, sklearn
2. **Load Data**: Read dataset and check for missing values
3. **Data Visualization**: Histograms and correlation matrix
4. **Data Preparation**: Split and scale features
5. **Model Training**: Train multiple linear regression model
6. **Evaluation**: Calculate RMSE and visualize residuals

## Results
- **Coefficients**:
  - `rm`: 3.70 (positive relationship with `medv`)
  - `lstat`: -4.63 (negative relationship with `medv`)
- **Model Performance**:
  - RMSE: 5.44

## Key Findings
- More rooms (`rm`) increase the median home value.
- Higher percentage of lower status population (`lstat`) decreases the median home value.
- RMSE indicates the model's predictions are reasonably accurate within the $5,000 range.
