## Wine Classification with Naive Bayes: Feature Engineering & Selection

### 1. Data Preprocessing & Safety

- **Missing Value Check:** Checked the dataset for null values using inputs.isna().any() to ensure data integrity before modeling.
- **Data Isolation:** Created a new variable `df1` (and subsequently `inputs`) to store selected features, ensuring the original `load_wine` dataframe remained untouched and available for reference.

### 2. Feature Selection Logic

- **Initial Selection:** Initially selected 5 key chemical indicators: `alcohol`, `malic_acid`, `total_phenols`, `color_intensity`, and `od280/od315_of_diluted_wines`.
- **Multicollinearity Detection:** Used the `.corr()` method to analyze relationships between features. Found a significant correlation ($r \approx 0.7$) between `total_phenols` and `od280/od315_of_diluted_wines`.
- **Refinement:** Dropped `total_phenols` to eliminate redundant information and prevent the model from "double-counting" related chemical signals.

### 3. Model Performance

- **Validation:** After refining the features to a final set of 4, used `cross_val_score` (cv=5) to test the model's stability.
- **Result:** GaussianNB achieved a stable and high average accuracy of **94.2%**, proving that a more concise, non-redundant feature set leads to a more robust classifier.
