"""
Interactive demo, the main page of the SAMueL machine learning
model explanation app.

Because a long app quickly gets out of hand,
try to keep this document to mostly direct calls to streamlit to write
or display stuff. Use functions in other files to create and
organise the stuff to be shown. In this app, most of the work is
done in functions stored in files named container_(something).py
"""
# ----- Imports -----
import streamlit as st

# For compatibility with combo app,
# add an extra bit to the path if we need to.
try:
    # Try importing something as though we're running this from the
    # same directory as the landing page, Introduction.py.
    from utilities_ml.fixed_params import page_setup
except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_lifetime_stroke.
    import sys
    sys.path.append('./streamlit_stroke_treatment_ml/')
    # Now the following import will work:
    from utilities_ml.fixed_params import page_setup

# Custom functions:
from utilities_ml.fixed_params import write_markdown_in_colour
from utilities_ml.inputs import set_up_sidebar
import utilities_ml.inputs
import utilities_ml.main_calculations
# Containers:
import utilities_ml.container_inputs
import utilities_ml.container_metrics
import utilities_ml.container_bars
import utilities_ml.container_waterfalls
import utilities_ml.container_combo_waterfall
import utilities_ml.container_results


def main():
    # ###########################
    # ##### START OF SCRIPT #####
    # ###########################
    page_setup()

    # Set up the link to the "Details" page:
    import sys
    if './streamlit_stroke_treatment_ml/' in sys.path:
        path_to_details = './🔍_Details:_Thrombolysis_decisions#'
    else:
        path_to_details = './Details#'

    # Background info
    # Title:
    st.markdown(
        '''
        # :crystal_ball: Thrombolysis decisions

        The SAMueL-2 model gives the probability of
        any stroke team thrombolysing any patient.
        '''
        )
    # Draw the image with the basic model summary.
    try:
        st.image('./utilities_ml/SAMueL2_model_wide.png')
    except (FileNotFoundError,
            st.runtime.media_file_storage.MediaFileStorageError):
        # Add an extra bit to the path for the combo app.
        st.image('./streamlit_stroke_treatment_ml/' +
                 'utilities_ml/SAMueL2_model_wide.png')
    st.markdown(
        '''
        We can use the same patient details
        for all stroke teams to compare the decisions of different teams.
        ''',
        help=''.join([
            '🔍 - [Which stroke teams are included?]',
            f'({path_to_details}which-teams-are-included)',
            '\n\n',
            '🔍 - [What do the probabilities mean?]',
            f'({path_to_details}what-do-the-probabilities-mean)'
        ])
        )
    st.markdown('### Data used to make the model')
    st.markdown(
        '''
        We use data sets called __🔮 Training data__ (110,000 patients)
        and __🔮 Testing data__ (10,000 patients)
        which have these properties:
        ''',
        help=''.join([
            '🔍 - [What data is used?]',
            f'({path_to_details}what-data-is-used)',
        ])
    )
    st.markdown(
        '''
        | | |
        | --- | --- |
        | ✨ Cleaned | ⏰ Onset time known |
        | 🚑 Ambulance arrivals | ⏳🩻 Onset to scan under 4 hours |
        | 👥 Teams with over 250 admissions | 🪚 Only 10 features |
        | 💉 Teams with at least 10 thrombolysis |  |

        '''
    )

    st.markdown('### How we categorise the results')
    cols_method = st.columns(3)
    with cols_method[0]:
        st.info(
            '''
            __Thrombolysis: yes or no?__

            If probability is at least 66.6%:  
            ✔️ would thrombolyse

            If probability is between 33.3% and 66.6%:  
            ❓ might thrombolyse

            If probability is below 33.3%:  
            ❌ would not thrombolyse
            '''
            )
    with cols_method[1]:
        st.markdown('')  # To match offset of info/error boxes
        st.markdown('__Benchmark teams__')
        st.markdown(
            '''
            These teams are more likely than average to give
            thrombolysis to most patients.
            ''',
            help=''.join([
                '🔍 - [What are benchmark teams?]',
                f'({path_to_details}what-are-benchmark-teams)',
                '\n\n',
                '🔍 - [How are they picked?]',
                f'({path_to_details}how-are-the-benchmark-teams-picked)'
                ])
            )
    with cols_method[2]:
        st.error(
            '''
            __Benchmark decision__

            Each benchmark team can pick one of:  
            ✔️ would thrombolyse  
            ❓ might thrombolyse  
            ❌ would not thrombolyse

            The overall benchmark decision is the
            option picked by the biggest number of benchmark teams.
            '''
            )

    # User inputs
    with st.sidebar:
        use_plotly_events, container_input_patient_details = (
            set_up_sidebar(path_to_details))

    st.markdown('-' * 50)
    st.markdown('## :abacus: Predictions for this patient')
    st.info(
        'The patient details can be viewed and changed in the left sidebar.',
        icon='ℹ️'
        )

    # Draw some empty containers on the page.
    # They'll appear in this order, but we'll fill them in another order.
    container_metrics = st.container()
    with container_metrics:
        st.markdown('### How many teams would thrombolyse this patient?')

    container_propensity = st.container()
    with container_propensity:
        st.markdown('### How treatable is this patient?')

    container_highlighted_summary = st.container()
    with container_highlighted_summary:
        # st.markdown('## ')  # Breathing room
        st.markdown('### What would your team do?')
        st.caption(
            '''
            To highlight stroke teams, select them in this box
            or click on them in the interactive bar chart.
            ''',
            help=''.join([
                '🔍 - [Why are the team names numbers?]'
                f'({path_to_details}why-are-teams-given-numbers)'
                '\n\n',
                '🔍 - [What is my team called?]'
                f'({path_to_details}what-is-my-team-called)'
                '\n\n',
                '🔍 - [How is the data anonymised?]'
                f'({path_to_details}how-is-the-data-anonymised)'
            ])
            )
        # Set up columns to put the highlighted teams' info in.
        # One of these columns will also contain the highlighted
        # teams input.
        cols_highlighted_summary = st.columns(4)
        with cols_highlighted_summary[0]:
            container_input_highlighted_teams = st.container()

    container_bar_chart = st.container()
    with container_bar_chart:
        st.markdown('### How likely is thrombolysis for each team?')
        st.caption('To see the team names, hover or click on a bar.')

    # ###########################
    # ########## SETUP ##########
    # ###########################
    from utilities_ml.inputs import setup_for_app
    from utilities_ml.fixed_params import n_benchmark_teams, \
        default_highlighted_team, display_name_of_default_highlighted_team
    (
        stroke_teams_list,
        highlighted_teams_input,
        X,
        headers_X,
        benchmark_df,
        benchmark_rank_list,
        inds_benchmark,
        highlighted_teams_list,
        hb_teams_list,
        hb_teams_input,
        user_inputs_dict,
        all_probs,
        all_reals,
        test_probs,
        test_reals
    ) = setup_for_app(
        container_input_highlighted_teams,
        container_input_patient_details,
        )

    # ##################################
    # ########## CALCULATIONS ##########
    # ##################################

    # Load in the model:
    from utilities_ml.fixed_params import ml_model_file
    model = utilities_ml.inputs.load_pretrained_model(ml_model_file)
    # Main useful array:
    sorted_results = utilities_ml.main_calculations.\
        predict_treatment(X, model, stroke_teams_list,
                          highlighted_teams_list, benchmark_rank_list,
                          hb_teams_list)


    # ###########################
    # ######### RESULTS #########
    # ###########################

    with container_metrics:
        # Print metrics for how many teams would thrombolyse:
        utilities_ml.container_metrics.main(sorted_results, n_benchmark_teams)

    with container_propensity:
        # How treatable is this patient:
        st.markdown(
            f'''
            The mean probability of thrombolysis across all teams is
            __{sorted_results["Probability_perc"].mean():.0f}%__.
            '''
            )

    line_str = ''
    with container_highlighted_summary:
        highlighted_teams_colours = \
            st.session_state['highlighted_teams_colours']
        # Print results for highlighted teams.
        cols = cols_highlighted_summary
        i = 1
        for t, team in enumerate(hb_teams_input):
            if 'enchmark' not in team:
                try:
                    col = cols[i]
                except IndexError:
                    # Start a new row:
                    i = 1
                    col = cols[i]
                    line_str = (
                        '''
                        --------------

                        '''
                        )
                i += 1

                df_here = sorted_results[sorted_results['HB team'] == team]
                colour_here = highlighted_teams_colours[team]
                if team == default_highlighted_team:
                    team = display_name_of_default_highlighted_team
                # else:
                    # team = 'Team ' + team
                with col:
                    if len(line_str) > 0:
                        st.markdown(line_str)
                    write_markdown_in_colour(
                        '<strong> Team ' + team + '</strong>',
                        colour=colour_here)
                    prob_here = df_here['Probability_perc'].values[0]
                    thromb_here = df_here['Thrombolyse_str'].values[0]
                    if 'Yes' in thromb_here:
                        emoji_here = '✔️ '
                        extra_str = ''
                    elif 'No' in thromb_here:
                        emoji_here = '❌ '
                        extra_str = 'would not '
                    else:
                        emoji_here = '❓ '
                        extra_str = 'might '
                    st.markdown(
                        f'''
                        Probability: {prob_here:.2f}%  
                        {emoji_here}{extra_str}thrombolyse
                        '''
                    )
                    # HTML horizontal rule is <hr> but appears in grey.

    with container_bar_chart:
        # Top interactive bar chart:
        utilities_ml.container_bars.main(
            sorted_results,
            hb_teams_input,
            use_plotly_events,
            default_highlighted_team,
            display_name_of_default_highlighted_team
            )


    # Model accuracy
    import pandas as pd
    import numpy as np

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


    all_probs_emoji, all_reals_emoji = make_emoji_lists(all_probs, all_reals)
    all_pr_dict = get_numbers_each_accuracy_band(all_probs, all_reals)

    test_probs_emoji, test_reals_emoji = make_emoji_lists(test_probs, test_reals)
    test_pr_dict = get_numbers_each_accuracy_band(test_probs, test_reals)


    st.write('### How accurate is the model?')

    all_n_total = len(all_probs)
    st.write(f'There are {all_n_total} patients in total in the test data.')
    if all_n_total > 0:
        write_accuracy(all_pr_dict, all_n_total)

    st.write('--------')
    test_n_total = len(test_probs)
    st.write(f'There are {test_n_total} patients like this in the test data.')
    if test_n_total > 0:
        write_accuracy(test_pr_dict, test_n_total)

    # import pandas as pd
    # df = pd.DataFrame(
    #     np.stack((test_probs_emoji, test_reals_emoji), axis=-1),
    #     columns=['Predicted', 'Real']
    # )
    # st.write(df)

    st.write('### Uncertainty')    
    # Uncertainty:
    df_std = pd.read_csv(f'./data_ml/shap_std.csv')
    # Pick out just the teams:
    df_std_teams = df_std[df_std['feature'].str.contains('team')]
    
    # What are +/- values for each of the features?
    def get_std_from_df(df, col, value):
        # Filter this feature only:
        df_col = df[df['feature'] == col]
        try:
            df_col['feature_value'] = df_col['feature_value'].astype(float)
        except:
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
                category_bounds = [float(t.replace('+','-').split('-')[0]) for t in categories]
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

    # Convert values to probability:
    # y_offset = 0.3926216513057263
    x_offset = -0.85
    def convert_shap_logodds_to_prob(prob, x_offset):
        from scipy.special import expit
        return expit(prob + x_offset)
    
    # Calculate uncertainties for all teams:
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

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_uncert['rank'],
        y=df_uncert['mean_real_shap_prob'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=df_uncert['upper_limit_real_shap_prob'] - df_uncert['mean_real_shap_prob'],
            arrayminus=df_uncert['mean_real_shap_prob'] - df_uncert['lower_limit_real_shap_prob'],
        )
    ))
    fig.update_yaxes(range=[0.0, 1.0])
    st.plotly_chart(fig)


    st.write(df_uncert)
    


    # ----- The end! -----


if __name__ == '__main__':
    main()
