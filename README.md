## Overview
This project aims to develop a simple and interpretable machine learning model for predicting house prices using historical residential property data. The objective is to build a reliable model capable of estimating the value of new properties efficiently using a limited set of key structural features, while also demonstrating a complete and well-structured end-to-end modeling workflow.

The dataset contains $4,140$ housing records, with each property described by numerical attributes such as living area, number of bedrooms and bathrooms, number of floors, lot size, construction year, renovation status, and selected quality indicators. The target variable is the property sale price.

Multiple regression models were evaluated, including ``Linear Regression``, ``Decision Tree``, and ``Random Forest``, using performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared, along with residual diagnostics. The project emphasizes practical evaluation, interpretability, and thoughtful model selection in real-world predictive analytics.

## The Questions
1. Can house prices be predicted using a limited set of basic structural features?
2. How does a linear regression model perform compared to more complex models?
3. What do residuals and model diagnostics reveal about prediction errors and model limitations?

## Tools I Used
This project was developed using the following tools to support analysis, modeling, and presentation:

- **Python** - the primary language used for data exploration and predictive modeling  
  - **Pandas** for data manipulation, cleaning, and feature selection  
  - **Matplotlib** for data visualization and residual diagnostics
  - **NumPy** for numerical operations
  - **scikit-learn** for model training, evaluation, and comparison

- **Jupyter Notebooks** - for running experiments and documenting the analysis in a clear, step-by-step manner 

- **Visual Studio Code** - for writing, testing, and organizing Python scripts  

- **Git & GitHub** - for version control, experiment tracking, and project sharing

## Import & Clean Up Data
The required libraries were imported and the dataset was loaded for initial inspection. Basic exploratory checks were performed to understand data structure and identify quality issues. While most features contained complete values, the **condition column** included several missing (NaN) entries.

To address this, missing values in the **condition column** were imputed using the median, a robust strategy that preserves data while reducing the influence of outliers. This ensured completeness without introducing unnecessary bias.

```python
df.isnull().sum()
df['condition'] = df['condition'].fillna(df['condition'].median())
df.head()
```
View my notebook with detailed steps here: [house_prices_prediction.ipynb](house_prices_prediction\house_prices_prediction.ipynb).

## The Analysis
In this stage, the dataset was prepared for modeling by selecting the relevant input features and defining the target variable. Based on  exploratory analysis and domain relevance, the following structural features were chosen: ``living area`` (sqft_living), ``number of bedrooms``, ``number of bathrooms``, ``number of floors``, ``condition``, ``year built``, ``year renovated``, and ``view``. The property price was used as the target variable.

```python
X = df[['sqft_living', 'bedrooms', 'floors', 'condition', 'yr_built', 'yr_renovated', 'view', 'bathrooms']]
y = df['price']
```
The data was then split into training and testing subsets to evaluate model performance on unseen data. An 80/20 train–test split was applied using ``train_test_split`` from ``scikit-learn``, with the ``random_state parameter`` set to 42 to ensure reproducibility and consistent results across runs. This produced the variables X_train, X_test, y_train, and y_test, which were used in subsequent modeling and evaluation steps.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```
After splitting the dataset into training and testing subsets, a Linear Regression model was chosen as a baseline due to its simplicity, interpretability, and ability to reveal how individual features influence house prices. The model was instantiated using LinearRegression from scikit-learn and trained on the training data (X_train, y_train).

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)
```

During training, the model estimated coefficients for each feature by minimizing the difference between the predicted and actual target values. These coefficients represent how much the house price is expected to change with a one-unit increase in a given feature, assuming all other features remain constant. Once trained, the model was applied to the test set to generate predictions, which were then used to evaluate performance and assess how well the model generalizes to unseen data.

After training the linear regression model, predictions were generated on the test dataset and evaluated using standard regression metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared. These metrics quantify overall prediction error, average absolute deviation from true prices, and the proportion of variance in house prices explained by the model.

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
y_pred = model.predict(X_test)
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculate the R-squared value
r_squared = r2_score(y_test, y_pred)
# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
```
## Results
The model explains approximately **39%** of the variance in house prices, with an average prediction error of about $166,000. This is reasonable for a baseline model built using only a **limited set of structural features**, but it also highlights that many important pricing drivers such as ``location``, ``neighborhood characteristics``, and ``view quality`` are not included in the model.

Residual analysis shows that errors are generally centered around zero but become larger for higher-priced properties. The Actual vs. Predicted plot confirms that while the model follows the overall price trend, it consistently underestimates more expensive homes, indicating heteroscedasticity and potential non-linear relationships.

![actual_vs_prdicted_house_prices.png](images/actual_vs_prdicted_house_prices.png)  
*Linear regression captures the general price trend but underperforms for expensive properties*

The results indicate that larger homes tend to be more expensive, with living area being the strongest and most reliable predictor of price. Homes in better condition are also typically priced higher, reflecting perceived quality and maintenance level.

Interestingly, once overall size is controlled for, having more bedrooms does not necessarily increase price. In some cases, more bedrooms within the same space may mean smaller or less functional rooms, which can reduce attractiveness. Similarly, houses with more floors may be valued slightly lower, possibly due to preferences for simpler layouts and accessibility.

Overall, the findings suggest that how space is used matters more than the number of rooms, helping explain why the model produces the predictions it does.

## Insights:
1. Can house prices be predicted using a limited set of basic structural features? Yes, to a meaningful extent. Using only basic property characteristics such as living area, number of bedrooms, floors, condition, and construction/renovation year, the model is able to capture overall pricing trends and explains about 39% of the variation in house prices. While this feature set is not sufficient for highly accurate individual price predictions, it provides a solid and interpretable baseline model capable of producing consistent estimates.

2. How does an interpretable linear regression model perform compared to more complex tree-based models? The Linear Regression model outperformed both the Decision Tree and Random Forest models when using default parameters. Despite being simpler, it generalized better to unseen data and produced more stable predictions, demonstrating that higher model complexity does not automatically lead to better performance, especially when relevant location and quality-related features are missing.

3. What do residuals and model diagnostics reveal about prediction errors and model limitations? Residual analysis shows that prediction errors increase for more expensive homes and are not evenly distributed across the price range. This indicates heteroscedasticity and suggests that important pricing factors, such as neighborhood location, view quality and premium property characteristics are not captured in the current dataset. These diagnostics clearly show where the model performs reliably and where predictions should be interpreted with caution.

## What I Learned
Through this project, I learned how to structure an end-to-end machine learning workflow, starting from data preparation and feature selection to model training, evaluation, and interpretation. Working with real housing data reinforced the importance of clean inputs and thoughtful feature choices before applying any modeling technique.

I also gained practical insight into model evaluation beyond headline metrics. Comparing ``Linear Regression``, ``Decision Tree``, and ``Random Forest`` models showed that more complex models do not automatically deliver better results. In this case, linear regression provided a strong, interpretable baseline, while tree-based models suffered from overfitting due to limited feature richness and lack of tuning.

Finally, this project strengthened my ability to translate technical results into plain-English insights. By analyzing coefficients, residuals, and visual diagnostics, I learned how to explain model behavior, limitations, and business impact in a way that is understandable to non-technical stakeholders, an essential skill for applying data science in real-world decision-making.

## Challenges I Faced
One of the main challenges was working with a limited set of features, which restricted the performance of more complex models. This highlighted the risk of overfitting and the importance of feature richness over model complexity.

Another challenge was interpreting model performance beyond raw metrics. Understanding residual patterns, negative R² values, and why simpler models outperformed more advanced ones required careful analysis and reinforced the need for diagnostic visualizations.

# Conclusion
This project shows that a simple, well-designed model can deliver meaningful and interpretable results when data quality and feature selection are prioritized. Linear regression proved to be a strong, transparent baseline, while more complex models require tuning and richer features to add value.

Overall, the project emphasizes practical model evaluation, clear communication of results, and thoughtful model selection, key skills for applying data analysis and machine learning in real business contexts.