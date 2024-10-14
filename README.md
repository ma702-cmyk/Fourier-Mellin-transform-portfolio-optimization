# Fourier-Mellin-transform-portfolio-optimization
# Project Title

This repository contains the code and data used for portfolio optimization analysis. Below is an explanation of each file included in this project.

## Figures and Tables

- **fig3.eps**: This plot represents the stock data used in the portfolio optimization time series starting from April 1, 2013, to April 1, 2014.
- **Table 1**: Descriptive statistics on the stock prices.
- **fig4.eps**: Dendogram of the stock prices.
- **fig5.eps**: Average correlation of the samples grouped by correlation level.
- **fig6.eps**: First 10 principal components of the stock log returns.
- **fig7.eps**: Correlation heatmap of the principal components.
- **fig8.eps**: Cumulative returns of the principal components of the log returns.
- **fig9.eps**: Cumulative returns of the principal components of the log returns.
- **fig10.eps**: Wavelet transform of some of the principal components of log returns.
- **fig11.eps**: Performance evaluation: the profitability of VAR(1)-AutoML, CWT-CNN, and FM-LSTM.
- **Table 2**: MAE and MSE for the training set across the three methods.
- **Table 3**: MAE and MSE for the testing dataset across the three methods.

## Code and Data Files

- **d1.mat**: The data of 1421 stocks using MATLAB function from money.net site, stored as a MAT file.
- **main.m**: The main code that runs the three models and applies analysis (profitability, error, training process, testing process).
- **stack.m**: A helper function called in the main code to stack data to a 2D vector using a feature extraction approach to work with CNN and LSTM.
- **FM.m**: A helper function called in the main code to apply the Fourier-Millen transform to extract geometric features.
- **hipass_filter.m**: A helper function for FM.m, applied in the main code to apply a high-pass filter of 2-D vector.
- **error_analsis.m**: This code applies MAE and RMSE and plots them as a time series.
- **calculateMetrics_csv_1.m**: This code applies MAE and RMSE on VAR(1)-AutoML (denoted as M in the main code) and CWT-CNN (denoted as m in the main code), storing them as a CSV file.
- **calculateMetrics_csv_OO.m**: This code applies MAE and RMSE on FM-LSTM (denoted as O in the main code).
- **test.mat**: Contains all main results obtained from three models (training and testing) to save time of running the code again.
- **test1.mat**: Used to see the effect of selecting the data randomly on the models.

## Usage

Provide instructions on how to use the code, run the models, and analyze the results.

## License

Include details about the license.

## Contact

Information for contacting the author or contributing to the project.
