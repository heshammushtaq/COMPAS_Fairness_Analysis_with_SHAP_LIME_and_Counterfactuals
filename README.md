# COMPAS Analysis in Python

## Purpose of the Analysis

This notebook extends the COMPAS analysis in Python by adding interpretability and counterfactual explanation methods to the prediction model.

The goal of the analysis is to:

- load and prepare the COMPAS dataset
- train and evaluate the prediction model
- compute SHAP values on the test set
- produce a beeswarm summary plot
- generate waterfall plots for the highest-risk and lowest-risk defendant in each racial group
- run LIME on the same four individuals
- compare SHAP and LIME feature attributions
- generate at least one counterfactual explanation per individual using DiCE
- identify any counterfactuals that require changes to immutable features such as race or sex

This analysis helps examine not only model predictions, but also fairness, transparency, and governance concerns in a high-stakes setting.

## Python Libraries Used

The notebook uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `shap`
- `lime`
- `dice-ml`
- `IPython.display`
- `warnings`

### What Each Library Is Used For

- **pandas**: loading the dataset, filtering rows, and managing tables
- **numpy**: numerical operations and array-based calculations
- **matplotlib**: generating plots and figures
- **seaborn**: additional visualizations
- **scikit-learn**: preprocessing, train-test split, and evaluation utilities
- **xgboost**: training the prediction model
- **shap**: computing SHAP values and producing beeswarm and waterfall plots
- **lime**: creating local explanations for selected individuals
- **dice-ml**: generating counterfactual explanations
- **IPython.display**: improving notebook output display
- **warnings**: suppressing unnecessary warning messages

## Instructions for Reproducing the Results

The Python code for this project is written in **Google Colab**.

To reproduce the results:

1. Open the notebook in Google Colab.
2. Upload or mount the COMPAS dataset file used in the analysis.
3. Make sure the required libraries are installed. If needed, run:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime dice-ml ipython
4. Run the notebook cells from top to bottom in order.

## Results
The analysis produced both predictive and interpretability results on the COMPAS dataset.

SHAP values were computed on the test set to explain the model’s predictions.
A SHAP beeswarm summary plot was created to show the most influential features across the test data.
SHAP waterfall plots were generated for the highest-risk and lowest-risk defendant in each racial group.
LIME explanations were produced for the same four individuals so that local feature attributions could be compared directly.
SHAP and LIME agreed on some of the main drivers of prediction, but they also differed in the ranking and strength of certain features.
These differences suggest that explanation results can vary depending on the method used, which is important in governance and decision review.
Counterfactual explanations were generated using DiCE for each selected individual.
The counterfactuals identified the minimal feature changes required to flip the model prediction.
Any counterfactual requiring changes to immutable features such as race or sex was flagged as problematic, since such changes are not realistic or ethically acceptable.

Overall, the notebook shows that SHAP, LIME, and counterfactual explanations are useful for understanding model behavior, but they should be interpreted carefully in fairness-sensitive applications.
