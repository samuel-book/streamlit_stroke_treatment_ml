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


def fudge_100_test_patients(pr_dict):
    n_total = np.sum(list(pr_dict.values()))

    # First try the easy way.
    copy_dict = {}
    for key, val in zip(pr_dict.keys(), pr_dict.values()):
        # The double rounding looks stupid but is more likely to result in
        # exactly 100 patients. Rounding directly to 0d.p. sometimes gives
        # 99 patients or 101 patients.
        # Add 1e-5 to make 0.5 round up to 1.0 instead of down to 0.0.
        copy_dict[key] = np.round(1e-5 + np.round(100.0 * val / n_total, 1), 0).astype(int)
    # Check if this adds up to 100:
    sum_int = np.sum(list(copy_dict.values()))
    if sum_int == 100:
        return copy_dict
    else:
        # Have to do the longer way.
        pass

    # Fudge 100 patients exactly.
    # Initially take just the integer parts of the numbers once
    # they've been scaled from n_total to 100.
    # Then add or subtract from the integers until the sum is 100.
    # Start by adding to numbers with large fractional parts
    # or subtracting from numbers with small fractional parts.

    arr = []
    for key, val in zip(pr_dict.keys(), pr_dict.values()):
        # Get a proportion out of 100:
        v = 100.0 * val / n_total
        # Store the integer part of v (int(v)),
        # the bit after the decimal point (v%1),
        # and values to track how many times the integer part
        # has been fudged upwards and downwards.
        row = [key, int(v), v % 1, 0, 0]
        arr.append(row)
    arr = np.array(arr, dtype=object)

    # Cut off this process after 20 loops.
    loops = 0
    sum_int = np.sum(arr[:, 1])

    while loops < 20:
        if sum_int < 100:
            # Pick out the values that have been added to
            # the fewest times.
            min_change = np.min(arr[:, 3])
            inds_min_change = np.where(arr[:, 3] == min_change)
            # Of these, pick out the value with the largest
            # fractional part.
            largest_frac = np.max(arr[inds_min_change, 2])
            ind_largest_frac = np.where(
                (arr[:, 2] == largest_frac) &
                (arr[:, 3] == min_change)
                )
            if len(ind_largest_frac[0]) > 1:
                # Arbitrarily pick the first if multiple options.
                ind_largest_frac = ind_largest_frac[0][0]
            # Add one to the integer part of this value
            # and record the change in column 3.
            arr[ind_largest_frac, 1] += 1
            arr[ind_largest_frac, 3] += 1
        elif sum_int > 100:
            # Pick out the values that have been subtracted from
            # the fewest times.
            min_change = np.min(arr[:, 4])
            inds_min_change = np.where(arr[:, 4] == min_change)
            # Of these, pick out the value with the smallest
            # fractional part.
            smallest_frac = np.min(arr[inds_min_change, 2])
            ind_smallest_frac = np.where(
                (arr[:, 2] == smallest_frac) &
                (arr[:, 4] == min_change)
                )
            if len(ind_smallest_frac[0]) > 1:
                # Arbitrarily pick the first if multiple options.
                ind_smallest_frac = ind_smallest_frac[0][0]
            # Subtract one from the integer part of this value
            # and record the change in column 3.
            arr[ind_smallest_frac, 1] -= 1
            arr[ind_smallest_frac, 4] += 1
        sum_int = np.sum(arr[:, 1])
        if sum_int == 100:
            loops = 20
        else:
            loops += 1

    copy_dict = {}
    for i in range(arr.shape[0]):
        key = arr[i, 0]
        copy_dict[key] = arr[i, 1]

    return copy_dict


def find_accuracy(pr_dict):
    n_total = np.sum(list(pr_dict.values()))

    n_true_pos = pr_dict['yy'] + pr_dict['myy']
    n_true_neg = pr_dict['nn'] + pr_dict['mnn']

    acc = 100.0 * (n_true_pos + n_true_neg) / n_total
    return acc


def write_confusion_matrix(pr_dict):

    n_yy = pr_dict['yy'] + pr_dict['myy']
    n_yn = pr_dict['yn'] + pr_dict['myn']
    n_ny = pr_dict['ny'] + pr_dict['mny']
    n_nn = pr_dict['nn'] + pr_dict['mnn']

    table = np.array([
        [f"{n_yy} ({pr_dict['yy']} ✔️, {pr_dict['myy']} ❓✔️)",
         f"{n_ny} ({pr_dict['ny']} ❌, {pr_dict['mny']} ❓❌)"],
        [f"{n_yn} ({pr_dict['yn']} ✔️, {pr_dict['myn']} ❓✔️)",
         f"{n_nn} ({pr_dict['nn']} ❌, {pr_dict['mnn']} ❓❌)"]
    ], dtype=object)

    table = np.array([
        [f"{n_yy}",
         f"{n_ny}"],
        # [f"({pr_dict['yy']} ✔️, {pr_dict['myy']} ❓✔️)",
        #  f"({pr_dict['ny']} ❌, {pr_dict['mny']} ❓❌)"],
        [f"({pr_dict['yy']} ✔️)", f"({pr_dict['ny']} ❌)"],
        [f"({pr_dict['myy']} ❓✔️)", f"({pr_dict['mny']} ❓❌)"],
        [f"{n_yn}",
         f"{n_nn}"],
        # [f"({pr_dict['yn']} ✔️, {pr_dict['myn']} ❓✔️)",
        #  f"({pr_dict['nn']} ❌, {pr_dict['mnn']} ❓❌)"]
        [f"({pr_dict['yn']} ✔️)", f"{pr_dict['mnn']} ❓❌)",],
        [f"({pr_dict['myn']} ❓✔️)", f"{pr_dict['nn']} ❌)"]
    ], dtype=object)

    df = pd.DataFrame(
        table,
        columns=['Predict ✔️', 'Predict ❌']
    )
    # df['Actual'] = ['Real ✔️', 'Real ✔️', 'Real ❌', 'Real ❌']
    # df = df.set_index('Actual')
    df = df.set_index([
        np.array(['Real ✔️', 'Real ✔️', 'Real ✔️', 'Real ❌', 'Real ❌', 'Real ❌']),
        np.array(['', '', '', '', '', ''])
    ])

    # Apply styles to colour the backgrounds:
    styles=[]
    # Change the background colour "background-color" of the box
    # and the colour of the text "color".
    # Use these colours...
    # colour_true = 'rgba(127, 255, 127, 0.2)'
    # colour_false = 'rgba(255, 127, 127, 0.2)'
    colour_true = 'rgba(0, 209, 152, 0.2)'
    colour_false = 'rgba(255, 116, 0, 0.2)'
    colour_blank = 'rgba(0, 0, 0, 0)'
    # ... in this pattern:
    colour_grid = [
        [colour_true, colour_false],
        [colour_true, colour_false],
        [colour_true, colour_false],
        [colour_false, colour_true],
        [colour_false, colour_true],
        [colour_false, colour_true]
    ]
    # Update each cell individually:
    for r, row in enumerate(colour_grid):
        for c, col in enumerate(row):
            colour = colour_grid[r][c]
            # Correct for multiindex shift
            # (who knows why this happens anyway)
            if r % 3 == 0:
                c += 1
            # elif r % 2 == 0:
            #     c -= 1
            # c+2 because HTML has one-indexing,
            # and the "first" header is the one above the index.
            # nth-child(2) is the header for the first proper column,
            # the one that's 0th in the list according to pandas.
            # (Working this out has displeased me greatly.)
            styles.append({
                'selector': f"tr:nth-child({r+1}) td:nth-child({c+2})",
                'props': [("background-color", f"{colour}")],
                        # ("color", "black")]
                })
    # Apply these styles to the pandas DataFrame:
    df_to_show = df.style.set_table_styles(styles)

    st.write(df_to_show.to_html(), unsafe_allow_html=True)


def write_confusion_matrix_maybe(pr_dict):
    table = np.array([
        [pr_dict['yy'], pr_dict['myy'], pr_dict['mny'], pr_dict['ny']],
        [pr_dict['yn'], pr_dict['myn'], pr_dict['mnn'], pr_dict['nn']]
    ])
    n_total = np.sum(table)

    # # Scale values to match 100 patients.
    # table = np.round(1e-5 + np.round(100.0 * table / n_total, 1), 0)
    # table = np.round(table, 0)
    # n_total = np.sum(table)
    # st.write(n_total)
    # st.write(table)


    df = pd.DataFrame(
        table,
        columns=['Predict ✔️', 'Predict ❓✔️', 'Predict ❓❌', 'Predict ❌']
    )
    df['Actual'] = ['Real ✔️', 'Real ❌']
    df = df.set_index('Actual')

    # Apply styles to colour the backgrounds:
    styles=[]
    # Change the background colour "background-color" of the box
    # and the colour of the text "color".
    # Use these colours...
    # colour_true = 'rgba(127, 255, 127, 0.2)'
    # colour_false = 'rgba(255, 127, 127, 0.2)'
    colour_true = 'rgba(0, 209, 152, 0.2)'
    colour_false = 'rgba(255, 116, 0, 0.2)'
    # ... in this pattern:
    colour_grid = [
        [colour_true, colour_true, colour_false, colour_false],
        [colour_false, colour_false, colour_true, colour_true]
    ]
    # Update each cell individually:
    for r, row in enumerate(colour_grid):
        for c, col in enumerate(row):
            colour = colour_grid[r][c]
            # c+2 because HTML has one-indexing,
            # and the "first" header is the one above the index.
            # nth-child(2) is the header for the first proper column,
            # the one that's 0th in the list according to pandas.
            # (Working this out has displeased me greatly.)
            styles.append({
                'selector': f"tr:nth-child({r+1}) td:nth-child({c+2})",
                'props': [("background-color", f"{colour}")],
                        # ("color", "black")]
                })
    # Apply these styles to the pandas DataFrame:
    df_to_show = df.style.set_table_styles(styles)

    st.table(df_to_show)


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

    return all_probs, all_reals, test_probs, test_reals, mask_number
