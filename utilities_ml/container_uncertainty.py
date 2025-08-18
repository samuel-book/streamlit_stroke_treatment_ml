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


def get_numbers_each_accuracy_band(test_probs: np.array,
                                   test_reals: np.array,
                                   allow_maybe=False,
                                   mn_min=0.333,
                                   my_max=0.666,
                                   ):
    """
    Find dict of values for confusion matrix.

    Find how many patients fall into each combination of
    predicted and real-life thrombolysis decision.
    The prediction may be:
    + yes - over 66%
    + maybe yes - 50 to 66%
    + maybe no - 33 to 50%
    + no - under 33%.

    Initials examples:
    + pr - predicted-real.
    + yy - yes predicted, yes real.
    + yn - yes predicted, no real.
    + myy - maybe yes predicted, yes real.
    etc.

    Predicted -->  |  y |  my |  mn |  n |
    ---------------+----+-----+-----+----+
            Real y | yy | myy | mny | ny |
            Real n | yn | myn | mnn | nn |

    WARNING - thresholds are hard-coded at the moment! 02/DEC/23

    Inputs
    ------
    test_probs - np.array. Model-predicted probability of thrombolysis
                 for each patient in the test data.
    test_reals - np.array. Real thrombolysis decision for each
                 patient in the test data.

    Returns
    -------
    pr_dict - dict. Contains numbers of patients in each part of
              the confusion matrix.
    """
    if allow_maybe:
        my_min = 0.50
        mn_max = 0.50
        # How many patients are in each prediction/real category?
        yy = len(np.where((test_probs > my_max) & (test_reals == 1))[0])
        yn = len(np.where((test_probs > my_max) & (test_reals == 0))[0])
        myy = len(np.where(
            (test_probs <= my_max) & (test_probs > my_min) & (test_reals == 1))[0])
        myn = len(np.where(
            (test_probs <= my_max) & (test_probs > my_min) & (test_reals == 0))[0])
        mny = len(np.where(
            (test_probs <= mn_max) & (test_probs >= mn_min) & (test_reals == 1))[0])
        mnn = len(np.where(
            (test_probs <= mn_max) & (test_probs >= mn_min) & (test_reals == 0))[0])
        ny = len(np.where((test_probs < mn_min) & (test_reals == 1))[0])
        nn = len(np.where((test_probs < mn_min) & (test_reals == 0))[0])
    else:
        yy = len(np.where((test_probs >= 0.5) & (test_reals == 1))[0])
        yn = len(np.where((test_probs >= 0.5) & (test_reals == 0))[0])
        ny = len(np.where((test_probs < 0.5) & (test_reals == 1))[0])
        nn = len(np.where((test_probs < 0.5) & (test_reals == 0))[0])
        # Set "maybe" data to zero:
        myy = 0
        myn = 0
        mny = 0
        mnn = 0

    pr_dict = {
        'yy': yy,
        'yn': yn,
        'myy': myy,
        'myn': myn,
        'mny': mny,
        'mnn': mnn,
        'ny': ny,
        'nn': nn,
    }
    return pr_dict


def fudge_100_test_patients(pr_dict: dict, allow_maybe=False):
    """
    Find confusion matrix values scaled to integers that sum to 100.

    First scale the matrix values to sum to 100 and then convert
    all values to integers. If the sum is now below 100 exactly,
    then add 1 to the value with the largest fractional part.
    Continue to add 1 to the value with the largest fractional part
    out of the set of values with fewest additions so far.
    Stop when the sum is 100 exactly.
    A similar process is coded in for the initial sum being higher
    than 100 and values being subtracted.

    Inputs
    ------
    pr_dict - dict. Predicted / Real decision dictionary. Each value
              is the number of patients with that combination of pr.

    Returns
    -------
    copy_dict - dict. The same as pr_dict but with values scaled to
                integer values that sum to 100.
    """
    # If we're not allowing "maybe" prediction then remove those
    # values from the dict:
    if allow_maybe:
        pass
    else:
        pr_dict_yn = dict(zip(
            pr_dict.keys(), [0 for i in range(len(pr_dict))]))
        keys_zeroed = []
        for k, v in pr_dict.items():
            if 'm' in k:
                keys_zeroed.append(k)
            else:
                pr_dict_yn[k] += v
        # Overwrite input dict:
        pr_dict = pr_dict_yn

    # How many patients are there to start with?
    n_total = np.sum(list(pr_dict.values()))

    # First try the easy way.
    # Scale everything to sum to 100 exactly and then take the integer
    # parts.
    copy_dict = {}
    for key, val in pr_dict.items():
        # The double rounding looks stupid but is more likely to result in
        # exactly 100 patients. Rounding directly to 0d.p. sometimes gives
        # 99 patients or 101 patients.
        # Add 1e-5 to make 0.5 round up to 1.0 instead of down to 0.0.
        copy_dict[key] = np.round(
            1e-5 + np.round(100.0 * val / n_total, 1), 0).astype(int)
    # Check if this adds up to 100:
    sum_int = np.sum(list(copy_dict.values()))
    if sum_int == 100:
        # Finished, don't do anything else.
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

    # Put all value parts and change trackers in here:
    arr = []
    for key, val in pr_dict.items():
        # Get a proportion out of 100:
        v = 100.0 * val / n_total
        # Store the value and bits to keep track of changes.
        row = [
            key,     # Name of this value.
            int(v),  # integer part of v.
            v % 1,   # fractional part of v.
            0,       # how many times we've added 1.
            0        # how many times we've subtracted 1.
            ]
        arr.append(row)
    arr = np.array(arr, dtype=object)

    # Cut off this process after 20 loops.
    loops = 0
    # What do the integer parts sum to?
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
            # and record the change in column 4.
            arr[ind_smallest_frac, 1] -= 1
            arr[ind_smallest_frac, 4] += 1

        # Check whether the values now sum to 100 as required.
        sum_int = np.sum(arr[:, 1])
        if sum_int == 100:
            # Stop the while loop.
            loops = 20
        else:
            # Restart the "while" loop.
            loops += 1

    # Make a new dictionary to store the new scaled values in.
    copy_dict = {}
    for i in range(arr.shape[0]):
        key = arr[i, 0]
        copy_dict[key] = arr[i, 1]
    # If necessary, add back in "maybe" placeholders:
    if allow_maybe:
        pass
    else:
        for k in keys_zeroed:
            copy_dict[k] = 0

    return copy_dict


def find_accuracy(pr_dict: dict):
    """
    Find accuracy of model from confusion matrix dict.

    Accuracy is number predicted correctly divided by total number.

    Inputs
    ------
    pr_dict - dict. Predicted / Real decision dictionary. Each value
              is the number of patients with that combination of pr.

    Returns
    -------
    acc - float. Accuracy as a percentage.
    """
    # Total number of patients in the confusion matrix:
    n_total = np.sum(list(pr_dict.values()))

    # Numbers of patients correctly predicted:
    n_true_pos = pr_dict['yy'] + pr_dict['myy']
    n_true_neg = pr_dict['nn'] + pr_dict['mnn']

    # Accuracy as a percentage.
    acc = 100.0 * (n_true_pos + n_true_neg) / n_total
    return acc


def write_confusion_matrix(pr_dict: dict, allow_maybe=False):
    """
    Style a confusion matrix and display with streamlit.

    Display the values in this table...

    Predicted -->   |  ✔️ | ❓✔️ |  ❌ | ❌ |
    ----------------+----+-----+-----+----+
            Real ✔️  | yy | myy | mny | ny |
            Real ❌ | yn | myn | mnn | nn |

    ... with correct cells (top left, lower right battenberg) in green
    and incorrect cells (top right, lower left battenberg) in red.
    """
    if allow_maybe:
        table = np.array([
            [pr_dict['yy'], pr_dict['myy'], pr_dict['mny'], pr_dict['ny']],
            [pr_dict['yn'], pr_dict['myn'], pr_dict['mnn'], pr_dict['nn']]
        ])
        cols = ['Predict ✔️', 'Predict ❓✔️', 'Predict ❓❌', 'Predict ❌']
    else:
        table = np.array([
            [pr_dict['yy'], pr_dict['ny']],
            [pr_dict['yn'], pr_dict['nn']]
        ])
        cols = ['Predict ✔️', 'Predict ❌']
    df = pd.DataFrame(
        table,
        columns=cols
    )
    df['Actual'] = ['Real ✔️', 'Real ❌']
    df = df.set_index('Actual')

    # Collect styles to colour the backgrounds:
    styles = []
    # Change the background colour "background-color" of the box.
    # Use these mostly-transparent seaborn colourblind colours...
    colour_true = 'rgba(0, 209, 152, 0.2)'   # greenish
    colour_false = 'rgba(255, 116, 0, 0.2)'  # reddish
    # ... in this pattern:
    if allow_maybe:
        colour_grid = [
            [colour_true, colour_true, colour_false, colour_false],
            [colour_false, colour_false, colour_true, colour_true]
        ]
    else:
        colour_grid = [
            [colour_true, colour_false],
            [colour_false, colour_true]
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
                'props': [("background-color", f"{colour}")]
                })
    # Apply these styles to the pandas DataFrame:
    df_to_show = df.style.set_table_styles(styles)

    st.table(df_to_show)


def find_similar_test_patients(user_inputs_dict: dict):
    """
    Find data on test patients similar to the input dict values.

    Use the same masks as were used to define the "similar patients"
    when creating the data files.

    Inputs
    ------
    user_inputs_dict - dict. Patient data dictionary for features
                       to run through the model, e.g. from input
                       from the streamlit app.

    Returns
    -------
    all_probs       - np.array. Thrombolysis predicted probability
                      for each patient.
    all_reals       - np.array. Thrombolysis yes/no for each patient.
    similar_probs   - np.array. Thrombolysis predicted probability
                      for each patient for similar patients only.
    similar_reals   - np.array. Thrombolysis yes/no for each patient
                      for similar patients only.
    all_n_train     - float. Number of patients in the training data.
    similar_n_train - float. Number of patients in the training data
                      that are similar to the input dict patient.
    """
    # What are the inds to look up similar test patients?
    # First check which masks are True and False for each
    # feature. The masks must match the ones used to create
    # the data files.
    masks_severity = [
        (user_inputs_dict['stroke_severity'] < 8),
        ((user_inputs_dict['stroke_severity'] >= 8) &
         (user_inputs_dict['stroke_severity'] <= 32)),
        (user_inputs_dict['stroke_severity'] > 32)
        ]
    masks_mrs = [
        ((user_inputs_dict['prior_disability'] == 0) |
         (user_inputs_dict['prior_disability'] == 1)),
        ((user_inputs_dict['prior_disability'] == 2) |
         (user_inputs_dict['prior_disability'] == 3)),
        ((user_inputs_dict['prior_disability'] == 4) |
         (user_inputs_dict['prior_disability'] == 5)),
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
        (user_inputs_dict['onset_to_arrival_time'] +
         user_inputs_dict['arrival_to_scan_time'] <= 4*60),
        (user_inputs_dict['onset_to_arrival_time'] +
         user_inputs_dict['arrival_to_scan_time'] > 4*60)
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
        'onset_scan': masks_onset_scan,
        'severity': masks_severity,
        'mrs': masks_mrs,
        'age': masks_age,
        'infarction': masks_infarction,
        'precise': masks_precise,
        'sleep': masks_sleep,
        'anticoag': masks_anticoag
    }

    # Store which mask is True for each feature.
    inds = {}
    for key, val in zip(masks.keys(), masks.values()):
        for i, m in enumerate(val):
            if m == 1:
                inds[key] = i

    # Which mask number in the data file
    # has True for all of these masks?
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
    # Limit to just "similar patients" (with this mask number):
    mask = (df_all_accuracy['mask_number'] == mask_number)
    similar_probs = all_probs[mask]
    similar_reals = all_reals[mask]

    # How many patients like this were in the training data?
    # Import training data group sizes.
    df_training_groups = pd.read_csv(f'{dir}data_ml/train_group_sizes.csv')
    all_n_train = df_training_groups['number_of_patients'].sum()
    mask = (df_training_groups['mask_number'] == mask_number)
    similar_n_train = (
        df_training_groups[mask]['number_of_patients'].values[0])

    return (all_probs, all_reals,
            similar_probs, similar_reals,
            all_n_train, similar_n_train)
