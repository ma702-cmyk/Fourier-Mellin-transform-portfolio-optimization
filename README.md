# Fourier-Mellin-transform-portfolio-optimization
# Project Title

This repository contains the code and data used for portfolio optimization analysis. Below is an explanation of each file included in this project.

## Figures and Tables

- **fig1.png**: This plot represents the stock data used in the portfolio optimization time series starting from April 1, 2013, to April 1, 2014.
- **fig2.png**: Dendogram of the stock prices.
- **fig3.png-fig4.png**: Average correlation of the samples grouped by correlation level.
- **fig5.png**: First 10 principal components of the stock log returns.
- **fig6.png**: Correlation heatmap of the principal components.
- **fig7.png**: Cumulative returns of the principal components of the log returns.
- **fig8.png**: Cumulative returns of the principal components of the log returns.
- **fig9.png**: Wavelet transform of some of the principal components of log returns.
- **e.png**: Performance evaluation: the profitability of VAR(1)-AutoML, CWT-CNN, and FM-LSTM.


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
- **FM.m**, **transform_Image.m** and **hipass_filter.m** was taken from MATLAB Central File Exchange:
 Robinson Laundon (2024). Fourier Mellin Image Registration (https://www.mathworks.com/matlabcentral/fileexchange/19731-fourier-mellin-image-registration), MATLAB Central File Exchange. Retrieved October 17, 2024. 
## Usage

To run the models **VAR(1)-AutoML**, **CWT-CNN** and **FM-LSTM** you can use **main.m** file. The code depends on the helper functions **FM.m**, **hipass_filter.m**, **transform_Image.m**, **calculateMetrics_csv_1.m**, **calculateMetrics_csv_OO.m** and **error_analsis.m**.

## Code and Research

This code is associated with our research paper titled "[Investigating the dynamics and uncertainties in portfolio optimization using the Fourier-Millen Transform]." It serves as a foundation for the analyses presented in the study. We encourage further development and use for future research purposes.
## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Contact

My email: mh.ayedh@tu.edu.sa.
