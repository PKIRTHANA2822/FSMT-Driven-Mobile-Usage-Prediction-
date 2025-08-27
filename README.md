# Project title 
- FSMT-Driven Mobile Usage Prediction with Feature Engineering
  <img width="834" height="834" alt="mobile-s3" src="https://github.com/user-attachments/assets/c5e4b5c4-b0d7-4c27-949f-9677d312ec3b" />
# Project Objective
- Predict a continuous target (e.g., price, revenue, energy use, time-to-complete) from historical data to support decisions like pricing, forecasting, capacity planning, or prioritization.
- Success criteria: choose 1–2 metrics up front (e.g., R² ≥ 0.80, RMSE ≤ X) and a tangible business outcome (e.g., reduce manual effort by 30%).
# Why do this project?
- Forecasting: plan inventory/demand/resources.
- Optimization: set prices or budgets better.
- Automation: replace or assist manual estimation.
- What-if analysis: test the impact of changing inputs.
# Step-by-step Approach (high level)
- Load & sanity-check the Excel data.
- Split columns: ID, target y, features X (num/cat/date/text).
- EDA to understand shape, leakage, missingness, outliers, and relationships.
- Preprocess (impute, encode, scale).
# Feature selection (statistical + model-based).
- Feature engineering (transformations & interactions).
- Train baseline(s) + tuned model(s).
- Test on a holdout set; analyze residuals & errors.
- Package outputs (predictions, feature importance, saved model).
- Code Skeleton You Can Adapt
- Replace target_col and file path; add/drop steps as needed.
# Load Data & Quick Checks
path = "data.xlsx"
df = pd.read_excel(path)  # or pd.read_csv("data.csv")
print(df.shape)
df.head()
# Define target & drop obvious IDs
target_col = "y"  # <-- change me
id_like = [c for c in df.columns if re.search(r'id$|_id$|^id$', c, flags=re.I)]
X = df.drop(columns=[target_col] + id_like, errors="ignore")
y = df[target_col]
# Basic info
print(X.info())
print(X.isna().mean().sort_values(ascending=False).head(10))  # missingness top 10
# Train/Test Split (do this early to avoid leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Exploratory Data Analysis (EDA)
Univariate
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
# Distributions
X_train[num_cols].hist(figsize=(12,8), bins=25); plt.show()
for c in cat_cols[:5]:
    X_train[c].value_counts(dropna=False).head(20).plot.bar(title=c); plt.show()
# Target distribution
y_train.plot.hist(bins=30, title="Target"); plt.show()
Bivariate (relationship to target)
# Numeric vs y
for c in num_cols[:6]:
    sns.scatterplot(x=X_train[c], y=y_train); plt.title(f"{c} vs y"); plt.show()
# Categorical vs y (boxplot through temporary merge)
tmp = X_train.assign(y=y_train)
for c in cat_cols[:4]:
    sns.boxplot(x=c, y="y", data=tmp); plt.title(f"{c} vs y"); plt.xticks(rotation=30); plt.show()
# Correlations & outliers
corr = pd.concat([X_train[num_cols], y_train], axis=1).corr()
sns.heatmap(corr, cmap="coolwarm", center=0); plt.title("Correlation"); plt.show()
# Feature Selection (choose 1–2 methods)
- Low variance filter
fs_variance = VarianceThreshold(threshold=0.0)
-  SelectKBest: f_regression or mutual info
fs_kbest = SelectKBest(score_func=f_regression, k=20)  # tune k
# alt: SelectKBest(mutual_info_regression, k=20)
- Model-based
est_for_fs = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
fs_model = SelectFromModel(est_for_fs, threshold="median")  # keep top 50%
- RFE
rfe = RFE(estimator=LinearRegression(), n_features_to_select=25)
- You’ll usually put one of these inside your main Pipeline after pre.
# Feature Engineering (examples)
- Log transform skewed numeric: np.log1p(x)
- Interactions/polynomials (with care): PolynomialFeatures(degree=2, interaction_only=True)
- Date parts: extract year/quarter/month/day/weekday, elapsed durations
- Binning high-cardinality categories; grouping rare levels as “Other”
- Domain ratios (e.g., amount_per_day = amount / days)
# Output (deliverables)
<img width="647" height="89" alt="Screenshot 2025-08-27 200757" src="https://github.com/user-attachments/assets/4568a964-a5b5-401d-b6e6-e5e4d2cf7775" />
