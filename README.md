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

In this stage, the dataset was prepared for modeling by selecting the relevant input features and defining the target variable. Based on  exploratory analysis and domain relevance, the following structural features were chosen: ``living area`` (sqft_living), ``number of bedrooms``, ``number of bathrooms``, and ``number of floors``. The property price was used as the target variable.

```python
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]
y = df['price']
```
The data was then split into training and testing subsets to evaluate model performance on unseen data. An 80/20 train–test split was applied using ``train_test_split`` from ``scikit-learn``, with the ``random_state parameter`` set to 42 to ensure reproducibility and consistent results across runs. This produced the variables X_train, X_test, y_train, and y_test, which were used in subsequent modeling and evaluation steps.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Check shape of splits
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
```
This approach establishes a clear and reproducible framework for comparing different regression models while minimizing data leakage and ensuring fair performance assessment.

After preparing the training and testing datasets, a Linear Regression model was selected as the baseline approach due to its simplicity, interpretability, and suitability for understanding relationships between features and house prices. An instance of the model was created using ``LinearRegression`` from scikit-learn and trained on the training data (X_train and y_train).

During training, the model learned the optimal coefficients for each input feature by minimizing the difference between predicted and actual house prices. These coefficients quantify the expected change in price associated with a one-unit change in each feature, holding all other variables constant. The trained model was then used to generate predictions on the test dataset for performance evaluation and diagnostic analysis.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)
```
After training the linear regression model, predictions were generated on the test dataset and evaluated using standard regression metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared. These metrics quantify overall prediction error, average absolute deviation from true prices, and the proportion of variance in house prices explained by the model.

## Results
The model explains approximately ``29%`` of the variance in house prices, with an average prediction error of around ``€180,000``. This is reasonable for a baseline model using only a few structural features, but it also shows that many pricing factors (such as location, view quality, neighborhood, and condition) are not captured in the dataset.

Residual analysis shows that errors are centered close to zero but increase in magnitude for higher-priced properties. The Actual vs. Predicted plot confirms that the model follows the general pricing trend but systematically underestimates expensive houses, indicating heteroscedasticity and non-linear effects.

![actual_vs_prdicted_house_prices.png](images/actual_vs_prdicted_house_prices.png)  
*Linear regression captures the general price trend but underperforms for expensive properties*

The model suggests that bigger homes tend to be more expensive, with living space being the strongest and most consistent driver of price. Homes with more bathrooms are also generally valued higher, as they offer greater comfort and functionality.

Interestingly, once the overall size of the home is taken into account, having more bedrooms does not necessarily increase the price. In some cases, more bedrooms within the same space may mean smaller rooms, which can make a home less attractive. Similarly, houses with more floors may be valued slightly lower, potentially reflecting buyer preferences for simpler layouts or easier accessibility.

Overall, these results show that how space is used matters more than how many rooms a home has, and they help explain why the model predicts prices the way it does.

## Insights:
1. Can house prices be predicted using a limited set of basic structural features? Yes, to a meaningful extent. Using only living area, number of bedrooms, bathrooms, and floors, the model was able to capture general pricing trends and explain approximately 29% of the variation in house prices. While these features are not sufficient for highly precise predictions, they provide a solid baseline for quick and consistent price estimation.

2. How does an interpretable linear regression model perform compared to more complex tree-based models? The linear regression model outperformed both the Decision Tree and Random Forest models when using default parameters. Despite its simplicity, it generalized better to unseen data and produced more stable results, demonstrating that increased model complexity does not automatically lead to better performance when **feature information is limited**.

3. What do residuals and model diagnostics reveal about prediction errors and model limitations? Residual analysis showed that prediction errors increase for higher-priced properties and are not evenly distributed across the price range. This indicates heteroscedasticity and suggests that important pricing factors, such as **location** and **property quality**, are missing from the feature set. These diagnostics highlight where the model is reliable and where predictions should be treated with caution.

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