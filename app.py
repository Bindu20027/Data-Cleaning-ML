import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

from clean import AutoClean, calculate_rmse
import base64
from pathlib import Path


def add_bg_design():
    bg_path = Path(__file__).parent / "assets" / "bg.jpg"

    if not bg_path.exists():
        st.error("Background image not found")
        return

    with open(bg_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        /* Main app background */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Content container */
        .block-container {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }}

        h1, h2, h3, p, label, span {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_design()

st.title("AutoClean - Performance-Driven Data Cleaning App")

st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.write("Original Data Head:", df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")

if df is not None:
    st.header("2. Configure Cleaning and ML Experiment")

    # Target Column Input
    all_columns = df.columns.tolist()
    target_column_name = st.selectbox(
        "Select your Target Column (y)",
        options=all_columns
    )

    # AutoClean Parameters Input
    st.subheader("AutoClean Strategy Configuration")
    mode_option = st.selectbox("AutoClean Mode", ['auto', 'manual'], index=1) # Default to manual for customization

    autoclean_params = {'mode': mode_option, 'verbose': False}

    if mode_option == 'manual':
        col1, col2 = st.columns(2)
        with col1:
            autoclean_params['missing_num'] = st.selectbox(
                "Handle Numerical Missing Values",
                ['knn', 'mean', 'median', 'most_frequent', 'linreg', 'delete', False],
                index=0 # Default to knn
            )
            autoclean_params['missing_categ'] = st.selectbox(
                "Handle Categorical Missing Values",
                ['most_frequent', 'logreg', 'knn', 'delete', False],
                index=0 # Default to most_frequent
            )
            autoclean_params['duplicates'] = st.selectbox(
                "Handle Duplicates",
                ['auto', False],
                index=0 # Default to auto
            )
        with col2:
            autoclean_params['outliers'] = st.selectbox(
                "Handle Outliers",
                ['winz', 'delete', False],
                index=0 # Default to winz
            )
            autoclean_params['extract_datetime'] = st.selectbox(
                "Extract Datetime Features (Granularity)",
                ['D', 'M', 'Y', 'h', 'm', 's', False],
                index=0 # Default to D
            )
            autoclean_params['encode_categ'] = st.selectbox(
                "Encode Categorical Features",
                [['onehot'], ['label'], ['auto'], False],
                format_func=lambda x: x[0].upper() if isinstance(x, list) else str(x),
                index=0 # Default to onehot
            )
            autoclean_params['outlier_param'] = st.slider("Outlier Multiplier", 0.5, 3.0, 1.5)

    if st.button("Run AutoClean Evaluation"):
        st.write("### Running Evaluation...")

        try:
            y = df[target_column_name]
            X = df.drop(columns=[target_column_name])
        except KeyError:
            st.error(f"Target column '{target_column_name}' not found.")
            st.stop()

        # Split BEFORE cleaning
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Data split: X_train {X_train.shape}, X_val {X_val.shape}")

        # AUTO-DETECT problem type
        if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 2:
            problem_type = 'regression'
            ml_model = LinearRegression()
            eval_metric = calculate_rmse
            metric_name = 'RMSE'
            st.info("Detected problem type: Regression")
        else:
            problem_type = 'classification'
            ml_model = LogisticRegression(max_iter=1000, solver='liblinear')
            eval_metric = accuracy_score
            metric_name = 'Accuracy'
            st.info("Detected problem type: Classification")

        # Strategies to test
        autoclean_strategies = [autoclean_params, {"mode": "auto", "verbose": False}]

        results = []

        for idx, params in enumerate(autoclean_strategies):
            st.write(f"--- Evaluating Strategy {idx+1}: {params['mode'].upper()} ---")

            # Combine X and y for cleaning to maintain consistent lengths during row deletions
            # Reset index to avoid issues with differing indices after split or prior operations
            train_df_combined = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
            val_df_combined = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)

            # Clean combined TRAIN independently — NO LEAKAGE
            cleaned_train_combined = AutoClean(input_data=train_df_combined.copy(), **params).output.copy()
            
            # Check if target column was removed during cleaning
            if target_column_name not in cleaned_train_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during cleaning for training set under strategy {params['mode']}. Skipping model training for this strategy.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            # Split back into features and target
            cleaned_train = cleaned_train_combined.drop(columns=[target_column_name])
            y_train_cleaned = cleaned_train_combined[target_column_name]

            # Clean combined VAL independently — NO LEAKAGE
            cleaned_val_combined = AutoClean(input_data=val_df_combined.copy(), **params).output.copy()
            
            # Check if target column was removed during cleaning
            if target_column_name not in cleaned_val_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during cleaning for validation set under strategy {params['mode']}. Skipping model training for this strategy.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            # Split back into features and target
            cleaned_val = cleaned_val_combined.drop(columns=[target_column_name])
            y_val_cleaned = cleaned_val_combined[target_column_name]

            # Drop non-numeric columns from features
            cleaned_train = cleaned_train.select_dtypes(include=np.number)
            cleaned_val = cleaned_val.select_dtypes(include=np.number)

            # Align columns between train/val (only for numeric features)
            common_cols = list(set(cleaned_train.columns) & set(cleaned_val.columns))
            
            # If no common numeric columns are left, this strategy is not viable
            if not common_cols:
                st.warning(f"No common numeric features found after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue

            cleaned_train = cleaned_train[common_cols]
            cleaned_val = cleaned_val[common_cols]

            # Safety check for feature columns (could be empty if all were dropped)
            if cleaned_train.empty or cleaned_val.empty or cleaned_train.shape[1] == 0 or cleaned_val.shape[1] == 0:
                metric_value = float('nan')
                st.warning(f"No usable features left after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                results.append({"params": params, metric_name: metric_value})
                continue
            
            # Ensure y_train_cleaned and y_val_cleaned are properly typed for models
            if problem_type == 'classification':
                y_train_cleaned = y_train_cleaned.astype(int)
                y_val_cleaned = y_val_cleaned.astype(int)
            
            # Check if the cleaned target variable became empty due to all rows being dropped
            if y_train_cleaned.empty or y_val_cleaned.empty:
                st.warning(f"Target variable became empty after cleaning for strategy {params['mode'].upper()}. Skipping model training.")
                metric_value = float('nan')
                results.append({"params": params, metric_name: metric_value})
                continue
            
            ml_model.fit(cleaned_train, y_train_cleaned) # Use cleaned y_train
            y_pred = ml_model.predict(cleaned_val)

            # y_val_cleaned and y_pred should now have consistent lengths due to the combined cleaning approach
            metric_value = eval_metric(y_val_cleaned, y_pred) # Use cleaned y_val
            results.append({"params": params, metric_name: metric_value})

        # ----------------------
        # Show Results
        # ----------------------
        st.header("3. Evaluation Results")

        best_val = float('inf') if metric_name == 'RMSE' else -float('inf')
        best_strategy = None

        for r in results:
            st.write(f"Strategy {r['params']} → {metric_name}: {r[metric_name]:.4f}")

            if metric_name == 'RMSE':
                if r[metric_name] < best_val:
                    best_val = r[metric_name]
                    best_strategy = r['params']
            else:
                if r[metric_name] > best_val:
                    best_val = r[metric_name]
                    best_strategy = r['params']

        if best_strategy:
            st.success(f"Best Strategy: {best_strategy}")
            st.success(f"Best {metric_name}: {best_val:.4f}")
        else:
            st.warning("No best strategy found (all strategies might have failed or resulted in NaN metric values).")

        # ----------------------
        # Final Cleaning + Download
        # ----------------------
        st.header("4. Cleaned Data Output")

        # Combine full X and y for final cleaning using the best strategy
        full_df_combined = pd.concat([df.drop(columns=[target_column_name]).copy().reset_index(drop=True), df[target_column_name].reset_index(drop=True)], axis=1)

        if best_strategy:
            final_cleaned_combined = AutoClean(input_data=full_df_combined, **best_strategy).output
            
            if target_column_name not in final_cleaned_combined.columns:
                st.warning(f"Target column '{target_column_name}' was removed during final cleaning with the best strategy. Cannot reconstruct original target. Outputting only cleaned features.")
                final_cleaned_df = final_cleaned_combined # Only features, potentially missing target
            else:
                final_clean_features = final_cleaned_combined.drop(columns=[target_column_name])
                final_clean_target = final_cleaned_combined[target_column_name]
                final_cleaned_df = pd.concat([final_clean_features, final_clean_target], axis=1)
        else:
            st.warning("No best strategy found (perhaps due to errors). Returning original data as cleaned output.")
            final_cleaned_df = df.copy() # Fallback to original if no best strategy

        st.write("Cleaned Data Preview:")
        st.dataframe(final_cleaned_df.head())

        st.download_button(
            label="Download Cleaned Dataset",
            data=final_cleaned_df.to_csv(index=False),
            file_name="cleaned_output.csv",
            mime="text/csv",
        )

