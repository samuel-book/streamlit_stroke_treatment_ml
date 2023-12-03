"""
Everything to do with uncertainty.

"""
import numpy as np
import pandas as pd
import streamlit as st


try:
    stroke_teams_test = pd.read_csv('./data_ml/stroke_teams.csv')
    dir = './'
except FileNotFoundError:
    dir = 'streamlit_stroke_treatment_ml/'


def make_emoji_lists(test_probs, test_reals):
    # WARNING - thresholds are hard-coded at the moment! 02/DEC/23
    test_probs_emoji = np.full(test_probs.shape, '')
    test_probs_emoji[test_probs > 0.66] = '✔️'
    test_probs_emoji[(test_probs <= 0.66) & (test_probs >= 0.33)] = '❓'
    test_probs_emoji[test_probs < 0.33] = '❌'

    test_reals_emoji = np.full(test_reals.shape, '')
    test_reals_emoji[test_reals == 1] = '✔️'
    test_reals_emoji[test_reals != 1] = '❌'
    return test_probs_emoji, test_reals_emoji


def get_numbers_each_accuracy_band(test_probs, test_reals):
    # WARNING - thresholds are hard-coded at the moment! 02/DEC/23
    pr_yy = len(np.where((test_probs > 0.66) & (test_reals == 1))[0])
    pr_yn = len(np.where((test_probs > 0.66) & (test_reals == 0))[0])
    pr_myy = len(np.where((test_probs <= 0.66) & (test_probs > 0.50) & (test_reals == 1))[0])
    pr_myn = len(np.where((test_probs <= 0.66) & (test_probs > 0.50) & (test_reals == 0))[0])
    pr_mny = len(np.where((test_probs <= 0.50) & (test_probs >= 0.33) & (test_reals == 1))[0])
    pr_mnn = len(np.where((test_probs <= 0.50) & (test_probs >= 0.33) & (test_reals == 0))[0])
    pr_ny = len(np.where((test_probs < 0.33) & (test_reals == 1))[0])
    pr_nn = len(np.where((test_probs < 0.33) & (test_reals == 0))[0])

    pr_dict = {
        'yy':pr_yy,
        'yn':pr_yn,
        'myy':pr_myy,
        'myn':pr_myn,
        'mny':pr_mny,
        'mnn':pr_mnn,
        'ny':pr_ny,
        'nn':pr_nn,
    }
    return pr_dict


def write_accuracy(pr_dict, n_total):
    df = pd.DataFrame(
        np.array([
            ['✔️', ' ✔️', 'Correct', f'{pr_dict["yy"]}'],
            ['❌', ' ❌', 'Correct',  f'{pr_dict["nn"]}'],
            ['❓✔️', ' ✔️', 'Correct',  f'{pr_dict["myy"]}'],
            ['❓❌', ' ❌', 'Correct',  f'{pr_dict["mnn"]}'],
            ['✔️', ' ❌', 'Wrong',  f'{pr_dict["yn"]}'],
            ['❌', ' ✔️', 'Wrong',  f'{pr_dict["ny"]}'],
            ['❓✔️', ' ❌', 'Wrong',  f'{pr_dict["myn"]}'],
            ['❓❌', ' ✔️', 'Wrong',  f'{pr_dict["mny"]}']
            ]),
        columns=['Predicted', 'Actual', 'Match?', 'Number'],
    )
    st.table(df)


    st.write(f'Confidently correct: {(pr_dict["yy"] + pr_dict["nn"])} patients: {(pr_dict["yy"] + pr_dict["nn"]) / n_total:.0%}')
    st.write(f'Unsure and correct: {(pr_dict["myy"] + pr_dict["mnn"])} patients: {(pr_dict["myy"] + pr_dict["mnn"]) / n_total:.0%}')
    st.write(f'Unsure and wrong: {(pr_dict["myn"] + pr_dict["mny"])} patients: {(pr_dict["myn"] + pr_dict["mny"]) / n_total:.0%}')
    st.write(f'Confidently wrong: {(pr_dict["yn"] + pr_dict["ny"])} patients: {(pr_dict["yn"] + pr_dict["ny"]) / n_total:.0%}')


def find_similar_test_patients(user_inputs_dict):
    # What are the inds to look up similar test patients?
    masks_severity = [
        (user_inputs_dict['stroke_severity'] < 8),
        ((user_inputs_dict['stroke_severity'] >= 8) & (user_inputs_dict['stroke_severity'] <= 32)),
        (user_inputs_dict['stroke_severity'] > 32)
        ]
    masks_mrs = [
        ((user_inputs_dict['prior_disability'] == 0) | (user_inputs_dict['prior_disability'] == 1)),
        ((user_inputs_dict['prior_disability'] == 2) | (user_inputs_dict['prior_disability'] == 3)),
        ((user_inputs_dict['prior_disability'] == 4) | (user_inputs_dict['prior_disability'] == 5)),
        ]
    masks_age = [
        (user_inputs_dict['age'] < 80),
        (user_inputs_dict['age'] >= 80)
        ]
    masks_infarction = [
        (user_inputs_dict['infarction'] == 0),
        (user_inputs_dict['infarction'] != 0)
        ]
    masks_onset_scan = [
        (user_inputs_dict['onset_to_arrival_time'] + user_inputs_dict['arrival_to_scan_time'] <= 4*60),
        (user_inputs_dict['onset_to_arrival_time'] + user_inputs_dict['arrival_to_scan_time'] > 4*60)
        ]
    masks_precise = [
        (user_inputs_dict['onset_time_precise'] == 0),
        (user_inputs_dict['onset_time_precise'] != 0)
        ]
    masks_sleep = [
        (user_inputs_dict['onset_during_sleep'] == 0),
        (user_inputs_dict['onset_during_sleep'] != 0)
        ]
    masks_anticoag = [
        (user_inputs_dict['anticoag'] == 0),
        (user_inputs_dict['anticoag'] != 0)
        ]

    masks = {
        'onset_scan':masks_onset_scan,
        'severity':masks_severity,
        'mrs':masks_mrs,
        'age':masks_age,
        'infarction':masks_infarction,
        'precise':masks_precise,
        'sleep':masks_sleep,
        'anticoag':masks_anticoag
    }

    # masks_names = list(masks.keys())
    # masks_lists = list(masks.values())

    inds = {}
    for key, val in zip(masks.keys(), masks.values()):
        for i, m in enumerate(val):
            if m == 1:
                inds[key] = i

    # Which mask number is this?
    df = pd.read_csv(f'{dir}data_ml/mask_numbers.csv')
    df_mask = df[
        (df['onset_scan_mask_number'] == inds['onset_scan']) &
        (df['severity_mask_number'] == inds['severity']) &
        (df['mrs_mask_number'] == inds['mrs']) &
        (df['age_mask_number'] == inds['age']) &
        (df['infarction_mask_number'] == inds['infarction']) &
        (df['precise_mask_number'] == inds['precise']) &
        (df['sleep_mask_number'] == inds['sleep']) &
        (df['anticoag_mask_number'] == inds['anticoag'])
    ]
    mask_number = df_mask['mask_number'].values[0]

    # Import all probabilities and thrombolysis:
    df_all_accuracy = pd.read_csv(f'{dir}data_ml/masks_probabilities.csv')
    all_probs = df_all_accuracy['predicted_probs']
    all_reals = df_all_accuracy['thrombolysis']

    # Mask for just this mask number:
    mask = (df_all_accuracy['mask_number'] == mask_number)
    test_probs = all_probs[mask]
    test_reals = all_reals[mask]

    return all_probs, all_reals, test_probs, test_reals


def get_std_from_df(df, col, value):
    """
    
    # What are +/- values for each of the features?
    """
    # Filter this feature only:
    df_col = df[df['feature'] == col]
    try:
        df_col['feature_value'] = df_col['feature_value'].astype(float)
    except ValueError:
        # Can't convert these values to float.
        pass
    # Find where the feature value matches input:
    if value in list(df_col['feature_value'].values):
        df_row = df_col[df_col['feature_value'] == value]
    else:
        # The exact value doesn't exist when the feature
        # is age, onset-to-arrival time, or arrival-to-scan time.
        # Temporarily? The exact value doesn't exist for
        # anticoagulant value missing.
        if 'anticoag' in col:
            df_row = df_col[
                (df_col['feature_value'] != 0) &
                (df_col['feature_value'] != 1)
                ]
        elif 'age' in col:
            # Need nearest multiple of 5 plus 2.5,
            # so options are 37.5, 42.5, 47.5, ... 92.5.
            if value < 37.5:
                value = 37.5
            elif value > 92.5:
                value = 92.5
            else:
                # Round to the nearest 5 and then subtract 2.5.
                # This means that age 40 goes to 37.5
                # and age 41 goes to 42.5.
                value = np.ceil(value / 5.0) * 5.0 - 2.5
            df_row = df_col[df_col['feature_value'] == value]
        elif 'time' in col:
            # Categories are string "0-29", "120-149", "150+".
            categories = np.unique(df_col['feature_value'])
            category_bounds = [float(t.replace('+','-').split('-')[0])
                               for t in categories]
            category_bounds = sorted(category_bounds)
            # Find which category the value falls into:
            bin = np.digitize(value, category_bounds) - 1
            category_here = categories[bin]
            df_row = df_col[df_col['feature_value'] == category_here]
        else:
            st.write('error: ', col, value)

    mean_shap = df_row['mean_shap'].values[0]
    std_shap = df_row['std_shap'].values[0]
    return mean_shap, std_shap


def get_this_patient_std_df(user_inputs_dict, df_std):
    std_col_dict = dict(
        arrival_to_scan_time='arrival_to_scan_time',
        infarction='infarction',
        stroke_severity='stroke_severity',
        onset_time_precise='precise_onset_known',
        prior_disability='prior_disability',
        anticoag='afib_anticoagulant',
        onset_to_arrival_time='onset_to_arrival_time',
        onset_during_sleep='onset_during_sleep',
        age='age'
    )
    arr = []
    for key, val in zip(user_inputs_dict.keys(), user_inputs_dict.values()):
        if key in list(std_col_dict.keys()):
            mean, std = get_std_from_df(df_std, std_col_dict[key], val)
            arr.append([std_col_dict[key], val, mean, std])
    df_std_this_patient = pd.DataFrame(
        arr,
        columns=['feature', 'feature_value', 'mean_shap', 'std_shap']
        )
    return df_std_this_patient


def make_shap_uncert(team, df_std_teams, df_std_this_patient, X):

    df_this_team = df_std_teams[
        (df_std_teams['feature'] == f'team_{team}') &
        (df_std_teams['feature_value'].astype(float) == 1)
        ]
    df_not_this_team = df_std_teams[
        (df_std_teams['feature'] != f'team_{team}') &
        (df_std_teams['feature_value'].astype(float) == 0)
        ]

    # Estimated SHAP value:
    # (THIS DOESN'T GIVE THE RIGHT ANSWER)
    mean_shap_this_team = df_this_team['mean_shap'].values[0]
    mean_shap_not_this_team = df_not_this_team['mean_shap'].sum()
    mean_shap_teams  = mean_shap_this_team + mean_shap_not_this_team
    # What is the total shap sum so far?
    # using mean values, not actual patient values for now... (why?)
    # (change this to the actual values)
    mean_shap_without_team = df_std_this_patient['mean_shap'].sum()
    # Combine:
    mean_shap = mean_shap_teams + mean_shap_without_team

    # Get actual SHAP values:
    # Data for this team:
    X_here = X[X[f'team_{team}'] == 1]
    # Get SHAP:
    from utilities_ml.inputs import load_explainer
    explainer = load_explainer()
    shap_values = explainer.shap_values(X_here)
    mean_real_shap = np.sum(shap_values)

    # Estimated uncertainties:
    uncert_shap_without_team = np.sqrt(np.sum([v**2.0 for v in df_std_this_patient['std_shap'].values]))
    all_std_teams = np.append(df_not_this_team['std_shap'].values, df_this_team['std_shap'].values[0])
    uncert_teams  = np.sqrt(np.sum([a**2.0 for a in all_std_teams ]))
    uncert_shap = np.sqrt(uncert_shap_without_team**2.0 + uncert_teams**2.0)
    return mean_real_shap, uncert_shap, mean_shap


def convert_shap_logodds_to_prob(prob, x_offset):
    from scipy.special import expit
    return expit(prob + x_offset)


def calculate_uncertainties_for_all_teams(stroke_teams_list, df_std_teams, df_std_this_patient, X, x_offset):
    arr = []
    for t, team in enumerate(stroke_teams_list):
        mean_real_shap, uncert_shap, mean_shap = make_shap_uncert(team, df_std_teams, df_std_this_patient, X)

        upper_limit_real_shap = mean_real_shap + uncert_shap
        lower_limit_real_shap = mean_real_shap - uncert_shap
        mean_real_shap_prob = convert_shap_logodds_to_prob(mean_real_shap, x_offset)
        upper_limit_real_shap_prob = convert_shap_logodds_to_prob(upper_limit_real_shap, x_offset)
        lower_limit_real_shap_prob = convert_shap_logodds_to_prob(lower_limit_real_shap, x_offset)

        upper_limit_shap = mean_shap + uncert_shap
        lower_limit_shap = mean_shap - uncert_shap
        mean_shap_prob = convert_shap_logodds_to_prob(mean_shap, x_offset)
        upper_limit_shap_prob = convert_shap_logodds_to_prob(upper_limit_shap, x_offset)
        lower_limit_shap_prob = convert_shap_logodds_to_prob(lower_limit_shap, x_offset)

        row = [
            mean_real_shap,
            uncert_shap,
            upper_limit_real_shap,
            lower_limit_real_shap,
            mean_real_shap_prob,
            upper_limit_real_shap_prob,
            lower_limit_real_shap_prob,
            mean_shap,
            upper_limit_shap,
            lower_limit_shap,
            mean_shap_prob,
            upper_limit_shap_prob,
            lower_limit_shap_prob,
        ]
        arr.append(row)

    
    df_uncert = pd.DataFrame(
        arr,
        columns=[
            'mean_real_shap',
            'uncert_shap',
            'upper_limit_real_shap',
            'lower_limit_real_shap',
            'mean_real_shap_prob',
            'upper_limit_real_shap_prob',
            'lower_limit_real_shap_prob',
            'mean_shap',
            'upper_limit_shap',
            'lower_limit_shap',
            'mean_shap_prob',
            'upper_limit_shap_prob',
            'lower_limit_shap_prob',
        ]
        )
    df_uncert = df_uncert.sort_values('mean_real_shap_prob', ascending=False)
    df_uncert['rank'] = np.arange(len(df_uncert))
    return df_uncert
