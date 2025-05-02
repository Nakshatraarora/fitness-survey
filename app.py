import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import statsmodels.api as sm
import warnings
import re
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress all warnings
warnings.filterwarnings('ignore')

# Streamlit page config
st.set_page_config(
    page_title="Fitness Influencer Survey Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define constants/mappings globally ---
TRUST_MAPPING = {
    'Do not trust at all': 0, 'low': 1, 'nutural': 2,
    'Somewhat trust': 3, 'high': 4, 'Completely trust': 5
}

# --- Define Column Names Globally ---
age_col = 'What is your age?'
gender_col = 'What is your gender?'
occupation_col = 'What is your current occupation'
social_media_freq_col = 'How often do you use social media?'
platforms_col = 'Which social media platforms do you use to follow fitness influencers? (Select all that apply)'
num_influencers_col = 'How many fitness influencers do you follow on social media?'
follow_reason_col = 'Why do you follow fitness influencers? (Select all that apply)'
engage_content_col = 'How often do you engage with fitness influencers’ content ?'
trust_col_orig = 'How much do you trust the nutrition advice given by fitness influencers?'
trust_encoded_col = 'trust_encoded'
change_habits_col = 'Have you ever changed your eating habits based on recommendations from a fitness influencer ?'
diet_changes_col = 'What kind of dietary changes have you made due to fitness influencers? (Select all that apply)'
research_claims_col = 'Do you research the health claims made by fitness influencers before following their advice?'
buy_products_col = 'How often do you buy food products or supplements promoted by fitness influencers?'
negative_effects_col = 'Have you ever experienced negative effects from following nutrition advice given by a fitness influencer?'
realistic_habits_col = 'Do you think fitness influencers promote realistic eating habits?'
qualified_col = 'Do you believe fitness influencers are qualified to give nutrition advice ?'
qual_importance_col = 'How important is it for a fitness influencer to have professional nutrition qualifications?'
unhealthy_trends_col = 'Do you think social media influences unhealthy eating trends (e.g., extreme dieting, unrealistic food restrictions)?'
stop_following_col = 'Have you ever stopped following a fitness influencer due to misleading health advice?'
verify_advice_col = 'How do you verify if a fitness influencer’s nutrition advice is credible?'
healthier_food_col = 'Have fitness influencers helped you develop a healthier relationship with food?'
transparency_col = 'Do you believe fitness influencers are transparent about their diet and food habits?'
meal_plan_col = 'Have you ever purchased a meal plan or diet guide created by a fitness influencer?'
spending_col = 'How much are you willing to spend on fitness influencer-recommended food products or supplements per month?'
buy_factors_col = 'What factors influence your decision to buy food products promoted by influencers? (Select all that apply)'
meal_prep_col = 'Do you prefer fitness influencers who share meal prep ideas and recipes?'
recommend_likelihood_col = 'How likely are you to recommend a diet or food product promoted by a fitness influencer to a friend?'
comments_col_orig = 'In your opinion, what could fitness influencers do to better promote healthy eating habits?'
comments_col_new = 'comments'
personal_experience_col = 'Do you have any personal experiences or concerns related to fitness influencers and nutrition advice?'

# --- Helper Functions ---

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load excel file and return a DataFrame."""
    try:
        df = pd.read_excel(path)
        df.columns = [re.sub(r'\s+', ' ', col.strip()) for col in df.columns]
        st.success(f"Successfully loaded data from {path}")
        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at '{path}'")
        return None
    except Exception as e:
        st.error(f"Error loading excel file: {e}")
        return None

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by handling missing values, data types, and categorical orders."""
    if df is None:
        return None

    df_cleaned = df.copy()
    st.info(f"Starting cleaning process with {len(df_cleaned)} rows.")

    # Check for expected columns and display actual columns if missing
    expected_cols = [
        age_col, gender_col, occupation_col, social_media_freq_col, platforms_col,
        num_influencers_col, follow_reason_col, engage_content_col, trust_col_orig,
        change_habits_col, diet_changes_col, research_claims_col, buy_products_col,
        negative_effects_col, realistic_habits_col, qualified_col, qual_importance_col,
        unhealthy_trends_col, stop_following_col, verify_advice_col, healthier_food_col,
        transparency_col, meal_plan_col, spending_col, buy_factors_col, meal_prep_col,
        recommend_likelihood_col, comments_col_orig, personal_experience_col
    ]
    missing_expected = [col for col in expected_cols if col not in df_cleaned.columns]
    if missing_expected:
        st.warning(f"Some expected columns are missing from the dataset: {', '.join(missing_expected)}")
        st.write("Actual columns in the dataset:", df_cleaned.columns.tolist())

    # Rename columns with potential extra spaces or typos (more robust matching)
    column_mapping = {}
    for col in df_cleaned.columns:
        col_clean = re.sub(r'\s+', ' ', col.strip()).lower()
        # Map variations of column names to the expected names
        if 'buy food products or supplements' in col_clean:
            column_mapping[col] = buy_products_col
        elif 'what is your age' in col_clean:
            column_mapping[col] = age_col
        elif 'what is your gender' in col_clean:
            column_mapping[col] = gender_col
        elif 'current occupation' in col_clean:
            column_mapping[col] = occupation_col
        elif 'how often do you use social media' in col_clean:
            column_mapping[col] = social_media_freq_col
        elif 'which social media platforms' in col_clean:
            column_mapping[col] = platforms_col
        elif 'how many fitness influencers' in col_clean:
            column_mapping[col] = num_influencers_col
        elif 'why do you follow fitness influencers' in col_clean:
            column_mapping[col] = follow_reason_col
        elif 'how often do you engage with fitness influencers' in col_clean:
            column_mapping[col] = engage_content_col
        elif 'trust the nutrition advice' in col_clean:
            column_mapping[col] = trust_col_orig
        elif 'changed your eating habits' in col_clean:
            column_mapping[col] = change_habits_col
        elif 'what kind of dietary changes' in col_clean:
            column_mapping[col] = diet_changes_col
        elif 'research the health claims' in col_clean:
            column_mapping[col] = research_claims_col
        elif 'negative effects from following nutrition advice' in col_clean:
            column_mapping[col] = negative_effects_col
        elif 'promote realistic eating habits' in col_clean:
            column_mapping[col] = realistic_habits_col
        elif 'qualified to give nutrition advice' in col_clean:
            column_mapping[col] = qualified_col
        elif 'professional nutrition qualifications' in col_clean:
            column_mapping[col] = qual_importance_col
        elif 'unhealthy eating trends' in col_clean:
            column_mapping[col] = unhealthy_trends_col
        elif 'stopped following a fitness influencer' in col_clean:
            column_mapping[col] = stop_following_col
        elif 'verify if a fitness influencer’s nutrition advice' in col_clean:
            column_mapping[col] = verify_advice_col
        elif 'healthier relationship with food' in col_clean:
            column_mapping[col] = healthier_food_col
        elif 'transparent about their diet' in col_clean:
            column_mapping[col] = transparency_col
        elif 'purchased a meal plan' in col_clean:
            column_mapping[col] = meal_plan_col
        elif 'spend on fitness influencer-recommended food products' in col_clean:
            column_mapping[col] = spending_col
        elif 'factors influence your decision to buy food products' in col_clean:
            column_mapping[col] = buy_factors_col
        elif 'prefer fitness influencers who share meal prep' in col_clean:
            column_mapping[col] = meal_prep_col
        elif 'recommend a diet or food product' in col_clean:
            column_mapping[col] = recommend_likelihood_col
        elif 'better promote healthy eating habits' in col_clean:
            column_mapping[col] = comments_col_orig
        elif 'personal experiences or concerns' in col_clean:
            column_mapping[col] = personal_experience_col

    df_cleaned = df_cleaned.rename(columns=column_mapping)

    # Standardize responses: lowercase, strip whitespace
    categorical_cols = [
        num_influencers_col, engage_content_col, change_habits_col, buy_products_col,
        realistic_habits_col, transparency_col, negative_effects_col, unhealthy_trends_col,
        stop_following_col, healthier_food_col, meal_plan_col, meal_prep_col,
        social_media_freq_col, research_claims_col, qualified_col, spending_col
    ]
    for col in categorical_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()

    # Drop rows missing required columns
    required_cols = [age_col, gender_col]
    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=required_cols)
    if len(df_cleaned) < original_rows:
        st.info(f"Dropped {original_rows - len(df_cleaned)} rows due to missing Age or Gender.")

    # Trust Column Processing
    if trust_col_orig in df_cleaned.columns:
        df_cleaned[trust_col_orig] = df_cleaned[trust_col_orig].astype(str).str.lower().str.strip()
        df_cleaned[trust_encoded_col] = df_cleaned[trust_col_orig].map(TRUST_MAPPING)
        if df_cleaned[trust_encoded_col].isna().any():
            st.warning(f"Some entries in '{trust_col_orig}' could not be mapped to numerical values. Setting unmapped values to 0 (no trust).")
            df_cleaned[trust_encoded_col].fillna(0, inplace=True)
    else:
        st.warning(f"Trust column '{trust_col_orig}' not found. Creating '{trust_encoded_col}' with default value 0 (no trust).")
        df_cleaned[trust_encoded_col] = 0  # Default to 0 instead of pd.NA

    # Comments Column
    if comments_col_orig in df_cleaned.columns:
        df_cleaned = df_cleaned.rename(columns={comments_col_orig: comments_col_new})
        df_cleaned[comments_col_new] = df_cleaned[comments_col_new].fillna('Not provided').replace(['.', 'NA', ''], 'Not provided')
    elif comments_col_new not in df_cleaned.columns:
        df_cleaned[comments_col_new] = 'Not provided'

    # Personal Experience Column
    if personal_experience_col in df_cleaned.columns:
        df_cleaned[personal_experience_col] = df_cleaned[personal_experience_col].fillna('Not provided').replace(['.', 'NA', ''], 'Not provided')

    # Age Column
    if age_col in df_cleaned.columns:
        df_cleaned[age_col] = df_cleaned[age_col].astype(str).str.strip()
        age_order = ['Below 18', '18-21', '22-25']
        present_categories = [cat for cat in age_order if cat in df_cleaned[age_col].unique()]
        if present_categories:
            df_cleaned[age_col] = pd.Categorical(df_cleaned[age_col], categories=present_categories, ordered=True)
        else:
            df_cleaned[age_col] = df_cleaned[age_col].astype('category')

    # Gender Column
    if gender_col in df_cleaned.columns:
        df_cleaned[gender_col] = df_cleaned[gender_col].astype('category')

    # Importance of Qualifications
    if qual_importance_col in df_cleaned.columns:
        df_cleaned[qual_importance_col] = pd.to_numeric(df_cleaned[qual_importance_col], errors='coerce')
        df_cleaned[qual_importance_col].fillna(df_cleaned[qual_importance_col].median(), inplace=True)

    # Recommend Likelihood
    if recommend_likelihood_col in df_cleaned.columns:
        df_cleaned[recommend_likelihood_col] = pd.to_numeric(df_cleaned[recommend_likelihood_col], errors='coerce')
        df_cleaned[recommend_likelihood_col].fillna(df_cleaned[recommend_likelihood_col].median(), inplace=True)

    # Other Categorical Columns
    other_categorical_cols = [
        platforms_col, follow_reason_col, diet_changes_col, buy_factors_col,
        verify_advice_col, occupation_col
    ]
    for col in other_categorical_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype('category')

    # Drop Duplicates
    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    if len(df_cleaned) < original_rows:
        st.info(f"Dropped {original_rows - len(df_cleaned)} duplicate rows.")

    # Drop unused columns
    for col in ['Timestamp', 'E-mail']:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])

    st.success(f"Cleaning finished. Returning {len(df_cleaned)} rows.")
    return df_cleaned

def encode_variables_for_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables for correlation and regression analysis."""
    df_encoded = df.copy()
    social_media_freq_map = {
        'never': 1, 'rarely': 2, 'occasionally': 3,
        'a few times a week': 4, 'weekly': 5, 'daily': 6
    }
    num_influencers_map = {
        'none': 0, '1-2': 1, '1-3': 2, '3-5': 3, '4-6': 4, 'more than 6': 5,
        '0': 0
    }
    engage_content_map = {
        'never': 1, 'once a week': 2, 'occasionally': 3,
        'few times a week': 4, 'weekly': 5, 'everyday': 6,
        'daily': 6
    }
    change_habits_map = {
        'no, never': 0, 'yes , sometimes': 1, 'yes, sometimes': 1,
        'yes , frequently': 2, 'frequently': 2, 'yes, frequently': 2,
        'yes, a lot of times': 3, 'no': 0, 'yes': 1
    }
    research_claims_map = {
        'never': 1, 'rarely': 2, 'sometimes': 3, 'always': 4
    }
    buy_products_map = {
        'never': 1, 'rarely': 2, 'occasionally': 3, 'frequently': 4, 'regularly': 5,
        'Never': 1, 'Rarely': 2, 'Occasionally': 3, 'Frequently': 4, 'Regularly': 5
    }
    negative_effects_map = {
        'no': 0, 'yes': 1, 'No': 0, 'Yes': 1
    }
    realistic_habits_map = {
        'never': 1, 'rarely': 2, 'sometimes': 3, 'yes, always': 4,
        'no': 1, 'yes': 4
    }
    qualified_map = {
        'no, most lack proper qualifications': 1, 'some are knowledgeable, but not all': 2,
        'some are qualified': 3, 'yes': 4, 'yes, most are experts': 5
    }
    unhealthy_trends_map = {
        'not at all': 1, 'not much': 2, 'somewhat': 3, 'strongly': 4, 'yes, a lot': 5
    }
    stop_following_map = {
        'no': 0, 'yes': 1, 'NO': 0, 'YES': 1
    }
    healthier_food_map = {
        'no': 0, 'yes': 1, 'NO': 0, 'YES': 1
    }
    transparency_map = {
        'no': 0, 'sometimes': 1, 'yes': 2, 'NO': 0, 'Sometimes': 1, 'YES': 2
    }
    meal_plan_map = {
        'no': 0, 'yes': 1, 'NO': 0, 'YES': 1
    }
    spending_map = {
        'nothing': 0, 'less than rs.500': 1, 'less than rs.1000': 2, 'more than rs.1000': 3,
        'rs.1000-2000': 4, 'rs.2000-5000': 5, 'more than rs.5000': 6
    }
    meal_prep_map = {
        'no': 0, 'yes': 1, 'NO': 0, 'YES': 1
    }

    mappings = {
        social_media_freq_col: social_media_freq_map,
        num_influencers_col: num_influencers_map,
        engage_content_col: engage_content_map,
        change_habits_col: change_habits_map,
        research_claims_col: research_claims_map,
        buy_products_col: buy_products_map,
        negative_effects_col: negative_effects_map,
        realistic_habits_col: realistic_habits_map,  
        qualified_col: qualified_map,
        unhealthy_trends_col: unhealthy_trends_map,
        stop_following_col: stop_following_map,
        healthier_food_col: healthier_food_map,
        transparency_col: transparency_map,
        meal_plan_col: meal_plan_map,
        spending_col: spending_map,
        meal_prep_col: meal_prep_map
    }

    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
            df_encoded[col].fillna(0, inplace=True)

    return df_encoded
def plot_correlation_matrix(df: pd.DataFrame, title: str):
    """Generate and display a correlation matrix heatmap."""
    corr_columns = [
        trust_encoded_col, change_habits_col, buy_products_col, healthier_food_col,
        social_media_freq_col, num_influencers_col, engage_content_col, research_claims_col,
        negative_effects_col, realistic_habits_col, qual_importance_col, unhealthy_trends_col,
        transparency_col, recommend_likelihood_col
    ]
    available_cols = [col for col in corr_columns if col in df.columns]
    missing_cols = [col for col in corr_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for correlation analysis: {', '.join(missing_cols)}")
        st.write("Available DataFrame columns:", df.columns.tolist())
        return

    df_encoded = encode_variables_for_correlation(df)
    df_corr = df_encoded[corr_columns].dropna()

    if df_corr.empty:
        st.warning("No data available for correlation matrix after removing missing values.")
        return

    corr_matrix = df_corr.corr().round(2)
    short_labels = [
        'Trust Score', 'Change Habits', 'Buy Products', 'Healthier Food',
        'Social Media', 'Num Influencers', 'Engage Content', 'Research Claims',
        'Negative Effects', 'Realistic Habits', 'Qual Importance', 'Unhealthy Trends',
        'Transparency', 'Recommend'
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1, center=0,
        cmap='viridis',  # Use viridis colormap
        square=True,
        annot=True,
        fmt=".2f",
        ax=ax,
        xticklabels=short_labels,
        yticklabels=short_labels
    )
    ax.grid(True, linestyle='--', alpha=0.7)  # Added gridlines
    ax.set_title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write("Correlation Matrix Table:")
    st.dataframe(corr_matrix)

def run_linear_regression(df: pd.DataFrame):
    """Run linear regression to predict monthly spending."""
    predictors = [num_influencers_col, engage_content_col, trust_encoded_col, change_habits_col, healthier_food_col]
    target = spending_col

    # Check if all columns exist
    missing_cols = [col for col in predictors + [target] if col not in df.columns]
    if missing_cols:
        st.error(
            f"Missing columns for linear regression: {', '.join(missing_cols)}\n"
            f"Expected columns: {', '.join(predictors + [target])}\n"
            f"Available columns in dataset: {', '.join(df.columns.tolist())}"
        )
        return

    # Encode variables
    df_encoded = encode_variables_for_correlation(df)
    X = df_encoded[predictors]
    y = df_encoded[target]

    # Drop rows with NaN in predictors or target
    data = pd.concat([X, y], axis=1).dropna()
    if data.empty:
        st.warning("No complete data available for linear regression.")
        return

    X = data[predictors]
    y = data[target]
    X = sm.add_constant(X)

    # Check for multicollinearity using Variance Inflation Factor (VIF)
    st.subheader("Multicollinearity Check (VIF)")
    vif_data = pd.DataFrame()
    vif_data["Predictor"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)
    st.write("**Note**: VIF > 5 indicates potential multicollinearity.")

    # Fit model
    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        st.error(f"Error fitting linear regression: {e}")
        return

    # Display results
    st.subheader("Linear Regression Results: Predicting Monthly Spending")
    st.write(f"**R²**: {model.rsquared:.3f} (proportion of variance explained)")
    st.write(f"**Adjusted R²**: {model.rsquared_adj:.3f}")

    # Create results table
    results_df = pd.DataFrame({
        'Predictor': ['Constant'] + predictors,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values,
        'p-value': model.pvalues.values,
        'Significant': ['Yes' if p < 0.05 else 'No' for p in model.pvalues]
    })
    results_df['Coefficient'] = results_df['Coefficient'].round(3)
    results_df['Std Error'] = results_df['Std Error'].round(3)
    results_df['p-value'] = results_df['p-value'].round(3)
    st.dataframe(results_df)

    # Plot significant predictors with shorter labels
    short_labels = {
        num_influencers_col: 'Num Influencers',
        engage_content_col: 'Engage Content',
        trust_encoded_col: 'Trust Score',
        change_habits_col: 'Change Habits',
        healthier_food_col: 'Healthier Food'
    }
    sig_results = results_df[results_df['Significant'] == 'Yes'][results_df['Predictor'] != 'Constant'].copy()
    if not sig_results.empty:
        sig_results['Predictor'] = sig_results['Predictor'].map(short_labels)
        num_predictors = len(sig_results)
        fig_height = max(2, num_predictors * 0.8)  # Dynamic height
        fig, ax = plt.subplots(figsize=(8, fig_height))
        sns.barplot(
            x='Coefficient',
            y='Predictor',
            data=sig_results,
            palette=sns.color_palette("viridis", n_colors=num_predictors),  # Use viridis palette
            width=0.5,
            ax=ax
        )
        ax.grid(True, axis='both', linestyle='--', alpha=0.7)  # Added gridlines on both axes
        ax.set_title('Significant Predictors of Monthly Spending', fontsize=12)
        ax.set_xlabel('Coefficient', fontsize=10)
        ax.set_ylabel('Predictor', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Interpretation
    st.markdown("**Interpretation**:")
    for _, row in results_df.iterrows():
        if row['Predictor'] != 'Constant' and row['Significant'] == 'Yes':
            short_name = short_labels.get(row['Predictor'], row['Predictor'])
            st.write(f"- **{short_name}**: A one-unit increase is associated with a {row['Coefficient']:.3f} unit change in monthly spending (p={row['p-value']:.3f}).")

def run_logistic_regression(df: pd.DataFrame):
    """Run logistic regression to predict meal plan purchase."""
    predictors = [num_influencers_col, engage_content_col, trust_encoded_col, change_habits_col, healthier_food_col]
    target = meal_plan_col

    # Check if all columns exist
    missing_cols = [col for col in predictors + [target] if col not in df.columns]
    if missing_cols:
        st.error(
            f"Missing columns for logistic regression: {', '.join(missing_cols)}\n"
            f"Expected columns: {', '.join(predictors + [target])}\n"
            f"Available columns in dataset: {', '.join(df.columns.tolist())}"
        )
        return

    # Encode variables
    df_encoded = encode_variables_for_correlation(df)
    X = df_encoded[predictors]
    y = df_encoded[target]

    # Drop rows with NaN in predictors or target
    data = pd.concat([X, y], axis=1).dropna()
    if data.empty:
        st.warning("No complete data available for logistic regression.")
        return

    X = data[predictors]
    y = data[target]
    X = sm.add_constant(X)

    # Check for multicollinearity using Variance Inflation Factor (VIF)
    st.subheader("Multicollinearity Check (VIF)")
    vif_data = pd.DataFrame()
    vif_data["Predictor"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)
    st.write("**Note**: VIF > 5 indicates potential multicollinearity.")

    # Fit model
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        st.error(f"Error fitting logistic regression: {e}")
        return

    # Display results
    st.subheader("Logistic Regression Results: Predicting Meal Plan Purchase")
    st.write(f"**Pseudo R²**: {model.prsquared:.3f}")

    # Create results table
    results_df = pd.DataFrame({
        'Predictor': ['Constant'] + predictors,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values,
        'p-value': model.pvalues.values,
        'Odds Ratio': np.exp(model.params.values),
        'Significant': ['Yes' if p < 0.05 else 'No' for p in model.pvalues]
    })
    results_df['Coefficient'] = results_df['Coefficient'].round(3)
    results_df['Std Error'] = results_df['Std Error'].round(3)
    results_df['p-value'] = results_df['p-value'].round(3)
    results_df['Odds Ratio'] = results_df['Odds Ratio'].round(3)
    st.dataframe(results_df)

    # Plot significant predictors with shorter labels
    short_labels = {
        num_influencers_col: 'Num Influencers',
        engage_content_col: 'Engage Content',
        trust_encoded_col: 'Trust Score',
        change_habits_col: 'Change Habits',
        healthier_food_col: 'Healthier Food'
    }
    sig_results = results_df[results_df['Significant'] == 'Yes'][results_df['Predictor'] != 'Constant'].copy()
    if not sig_results.empty:
        sig_results['Predictor'] = sig_results['Predictor'].map(short_labels)
        num_predictors = len(sig_results)
        fig_height = max(2, num_predictors * 0.8)  # Dynamic height
        fig, ax = plt.subplots(figsize=(8, fig_height))
        sns.barplot(
            x='Odds Ratio',
            y='Predictor',
            data=sig_results,
            palette=sns.color_palette("viridis", n_colors=num_predictors),  # Use viridis palette
            width=0.5,
            ax=ax
        )
        ax.grid(True, axis='both', linestyle='--', alpha=0.7)  # Added gridlines on both axes
        ax.set_title('Significant Predictors of Meal Plan Purchase (Odds Ratios)', fontsize=12)
        ax.set_xlabel('Odds Ratio', fontsize=10)
        ax.set_ylabel('Predictor', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Interpretation
    st.markdown("**Interpretation**:")
    for _, row in results_df.iterrows():
        if row['Predictor'] != 'Constant' and row['Significant'] == 'Yes':
            short_name = short_labels.get(row['Predictor'], row['Predictor'])
            st.write(f"- **{short_name}**: A one-unit increase multiplies the odds of purchasing a meal plan by {row['Odds Ratio']:.3f} (p={row['p-value']:.3f}).")

def plot_word_cloud(text_series, title):
    """Generate and display a word cloud."""
    text_data = text_series[text_series.str.lower() != 'not provided']
    if text_data.empty:
        st.warning(f"No comments provided for word cloud.")
        return

    text = ' '.join(text_data.astype(str).tolist())
    if not text.strip():
        st.warning(f"No text content available for word cloud.")
        return

    stopwords = set(STOPWORDS)
    try:
        wc = WordCloud(
            stopwords=stopwords,
            background_color='white',
            width=800,
            height=400,
            collocations=False,
            colormap='viridis'  # Use viridis colormap for the word cloud
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
    except ValueError as ve:
        st.warning(f"Could not generate word cloud: {ve}")
    except Exception as e:
        st.error(f"Error in word cloud generation: {e}")

def main():
    st.title("🏋🏽🔥💪🏼🎧 Fitness Influencer Survey Dashboard")
    st.sidebar.header("Configuration")

    data_path = st.sidebar.text_input("excel file path", value="anuj (Responses).xlsx")
    if not data_path:
        st.error("Please provide a valid file path.")
        return

    df_raw = load_data(data_path)
    if df_raw is None:
        return

    if st.sidebar.checkbox("Show raw data before cleaning", False):
        st.subheader("Raw Survey Data (Before Cleaning)")
        st.dataframe(df_raw)
        st.write(f"Raw data shape: {df_raw.shape}")
        st.write("Raw data columns:", df_raw.columns.tolist())

    df = clean_data(df_raw)
    if df is None or df.empty:
        st.error("Dataframe is empty after cleaning or cleaning failed.")
        return

    if st.sidebar.checkbox("Show Cleaned Data", False):
        st.subheader("Cleaned Survey Data")
        st.dataframe(df)
        st.write(f"Cleaned data shape: {df.shape}")
        st.write("Cleaned data columns:", df.columns.tolist())

    # Key Metrics
    st.markdown("---")
    st.subheader("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Valid Responses", len(df))

    if age_col in df.columns and not df[age_col].isnull().all():
        most_frequent_age = df[age_col].mode().iloc[0] if not df[age_col].mode().empty else "N/A"
        col2.metric("Most Frequent Age Group", most_frequent_age)
    else:
        col2.metric("Most Frequent Age Group", "N/A")

    if trust_encoded_col in df.columns and df[trust_encoded_col].notna().any():
        col3.metric("Average Trust Score (0-5)", f"{df[trust_encoded_col].mean():.2f}")
    else:
        col3.metric("Average Trust Score (0-5)", "N/A")

    # Descriptive Stats
    st.markdown("---")
    st.subheader("📈 Descriptive Statistics")
    try:
        st.write(df.describe(include='all'))
    except Exception as e:
        st.warning(f"Could not generate descriptive statistics: {e}")

    # Gender Distribution
    st.markdown("---")
    st.subheader("👥 Gender Distribution")
    if gender_col in df.columns and df[gender_col].notna().any():
        gender_counts = df[gender_col].value_counts()
        if not gender_counts.empty:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            viridis_colors = sns.color_palette("viridis", n_colors=2)  # Get 2 colors from viridis
            ax1.pie(
                gender_counts,
                labels=gender_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 10},
                colors=viridis_colors  # Use viridis colors
            )
            ax1.axis('equal')
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig1.gca().add_artist(centre_circle)
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.warning("No data for gender distribution.")
    else:
        st.warning(f"Column '{gender_col}' not found.")

    # Trust Level Counts
    st.markdown("---")
    st.subheader("🤝 Trust Level Counts")
    if trust_col_orig in df.columns and df[trust_col_orig].notna().any():
        trust_counts = df[trust_col_orig].value_counts()
        if not trust_counts.empty:
            fig_trust, ax_trust = plt.subplots(figsize=(8, 5))
            trust_counts.reindex(TRUST_MAPPING.keys()).plot(
                kind='bar',
                ax=ax_trust,
                color=sns.color_palette("viridis", n_colors=len(trust_counts))  # Use viridis palette
            )
            ax_trust.grid(True, axis='y', linestyle='--', alpha=0.7)  # Added gridlines
            ax_trust.set_xlabel('Trust Level')
            ax_trust.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_trust)
            plt.close(fig_trust)
        else:
            st.warning("No data for trust level counts.")
    elif trust_encoded_col in df.columns and df[trust_encoded_col].notna().any():
        trust_mapping_rev = {v: k for k, v in TRUST_MAPPING.items()}
        trust_counts_encoded = df[trust_encoded_col].map(trust_mapping_rev).value_counts()
        fig_trust, ax_trust = plt.subplots(figsize=(8, 5))
        trust_counts_encoded.reindex(TRUST_MAPPING.keys()).plot(
            kind='bar',
            ax=ax_trust,
            color=sns.color_palette("viridis", n_colors=len(trust_counts_encoded))  # Use viridis palette
        )
        ax_trust.grid(True, axis='y', linestyle='--', alpha=0.7)  # Added gridlines
        ax_trust.set_xlabel('Trust Level')
        ax_trust.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_trust)
        plt.close(fig_trust)
    else:
        st.warning(f"Trust columns missing or empty.")

    # Age Group Distribution
    st.markdown("---")
    st.subheader("🎂 Age Group Distribution")
    if age_col in df.columns and df[age_col].notna().any():
        age_counts = df[age_col].value_counts().sort_index()
        if not age_counts.empty:
            fig_age, ax_age = plt.subplots(figsize=(8, 5))
            age_counts.plot(
                kind='bar',
                ax=ax_age,
                color=sns.color_palette("viridis", n_colors=len(age_counts))  # Use viridis palette
            )
            ax_age.grid(True, axis='y', linestyle='--', alpha=0.7)  # Added gridlines
            ax_age.set_xlabel('Age Group')
            ax_age.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_age)
            plt.close(fig_age)
        else:
            st.warning("No data for age group distribution.")
    else:
        st.warning(f"Column '{age_col}' not found.")

    # Trust by Gender
    st.markdown("---")
    st.subheader("🤝 Trust by Gender")
    if gender_col in df.columns and trust_col_orig in df.columns:
        plot_data = df[[gender_col, trust_col_orig]].dropna()
        if not plot_data.empty:
            fig_tg, ax_tg = plt.subplots(figsize=(6, 4))
            sns.countplot(
                x=gender_col,
                hue=trust_col_orig,
                data=plot_data,
                ax=ax_tg,
                hue_order=list(TRUST_MAPPING.keys()),
                palette=sns.color_palette("viridis", n_colors=len(TRUST_MAPPING))  # Use viridis palette
            )
            ax_tg.grid(True, axis='y', linestyle='--', alpha=0.7)  # Added gridlines
            ax_tg.set_xlabel('Gender')
            ax_tg.set_ylabel('Count')
            ax_tg.legend(title='Trust Level')
            plt.tight_layout()
            st.pyplot(fig_tg)
            plt.close(fig_tg)
        else:
            st.warning("Not enough data for Trust by Gender.")
    else:
        st.warning(f"Required columns missing.")

    # Word Cloud
    st.markdown("---")
    st.subheader("💬 Word Cloud of Comments")
    if comments_col_new in df.columns:
        plot_word_cloud(df[comments_col_new], "Suggestions for Promoting Healthy Eating")
    else:
        st.warning(f"Column '{comments_col_new}' not found.")

    # Trust by Occupation
    st.markdown("---")
    st.subheader("💼 Trust by Occupation")
    if occupation_col in df.columns and trust_encoded_col in df.columns:
        crosstab_data = df[[occupation_col, trust_encoded_col]].dropna()
        if not crosstab_data.empty:
            crosstab = pd.crosstab(crosstab_data[occupation_col], crosstab_data[trust_encoded_col], normalize='index')
            if not crosstab.empty:
                trust_mapping_rev = {v: k for k, v in TRUST_MAPPING.items()}
                crosstab.columns = crosstab.columns.map(trust_mapping_rev)
                ordered_cols = [TRUST_MAPPING[key] for key in TRUST_MAPPING]
                ordered_names = [trust_mapping_rev[label] for label in ordered_cols if label in trust_mapping_rev]
                crosstab = crosstab[[col for col in ordered_names if col in crosstab.columns]]
                fig_ct, ax_ct = plt.subplots(figsize=(10, max(6, len(crosstab) * 0.5)))
                crosstab.plot(
                    kind='barh',
                    stacked=True,
                    ax=ax_ct,
                    cmap='viridis'  # Use viridis colormap
                )
                ax_ct.grid(True, axis='x', linestyle='--', alpha=0.7)  # Added gridlines
                ax_ct.set_xlabel('Proportion of Trust Level within Occupation')
                ax_ct.set_ylabel('Occupation')
                ax_ct.legend(title='Trust Level', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_ct.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig_ct)
                plt.close(fig_ct)
            else:
                st.warning("Crosstab empty.")
        else:
            st.warning("Not enough data for Trust by Occupation.")
    else:
        st.warning(f"Required columns missing.")

    # Average Trust Score by Age Group
    st.markdown("---")
    st.subheader("📊 Average Trust Score by Age Group")
    if age_col in df.columns and trust_encoded_col in df.columns:
        trust_by_age = df.groupby(age_col, observed=False)[trust_encoded_col].mean().dropna()
        if not trust_by_age.empty:
            fig_age_trust, ax_age_trust = plt.subplots(figsize=(8, 5))
            viridis_single = sns.color_palette("viridis", n_colors=1)[0]  # Get a single color from viridis
            trust_by_age.plot(
                kind='line',
                marker='o',
                ax=ax_age_trust,
                color=viridis_single  # Use a single viridis color
            )
            ax_age_trust.grid(True, axis='both', linestyle='--', alpha=0.7)  # Added gridlines
            ax_age_trust.set_title("Trend in Average Trust Score across Age Groups")
            ax_age_trust.set_xlabel("Age Group")
            ax_age_trust.set_ylabel("Average Trust Score (0-5)")
            ax_age_trust.set_ylim(bottom=0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_age_trust)
            plt.close(fig_age_trust)
        else:
            st.warning("No data for trust score by age group.")
    else:
        st.warning(f"Required columns missing.")

    # Correlation Analysis
    st.markdown("---")
    st.subheader("🔗 Correlation Analysis")
    st.markdown("Explore relationships between trust, engagement, and behavioral variables.")
    if st.checkbox("Show Correlation Matrix Heatmap", False):
        plot_correlation_matrix(df, "Correlation Matrix of Fitness Survey Variables")

    # Regression Analysis
    st.markdown("---")
    st.subheader("📉 Regression Analysis")
    st.markdown("Analyze how influencer engagement predicts shopping behavior.")
    if st.checkbox("Show Linear Regression Results", False):
        run_linear_regression(df)
    if st.checkbox("Show Logistic Regression Results", False):
        run_logistic_regression(df)

if __name__ == '__main__':
    main()