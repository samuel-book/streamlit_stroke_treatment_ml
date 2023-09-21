import streamlit as st
import numpy as np

import utilities_ml.main_calculations
import utilities_ml.inputs
import utilities_ml.container_combo_waterfall
import utilities_ml.container_waterfalls

from utilities_ml.fixed_params import draw_sneaky_bar, \
    plain_str, bench_str, model_version

# For compatibility with combo app,
# add an extra bit to the path if we need to.
try:
    # Try importing something as though we're running this from the
    # same directory as the landing page, Introduction.py.
    from utilities_ml.fixed_params import page_setup
    p = 'pages/text_for_pages/'

except ModuleNotFoundError:
    # If the import fails, add the landing page directory to path.
    # Assume that the script is being run from the directory above
    # the landing page directory, which is called
    # streamlit_lifetime_stroke.
    import sys
    sys.path.append('./streamlit_stroke_treatment_ml/')
    # Now the following import will work:
    from utilities_ml.fixed_params import page_setup



from utilities_ml.inputs import write_text_from_file

def main():
    page_setup()

    import sys
    if './streamlit_stroke_treatment_ml/' in sys.path:
        p = 'streamlit_stroke_treatment_ml/pages/text_for_pages/'
    else:
        p = 'pages/text_for_pages/'

    if './streamlit_stroke_treatment_ml/' in sys.path:
        path_to_details = './🔍_Details:_Thrombolysis_decisions#'
    else:
        path_to_details = './Details#'

    st.warning('Work in progress', icon='⚠️')

    # st.markdown('# 🗃️ 🗂️ 📁 ')

    # thing = st.toggle('What dis?')
    # if thing:
    #     pass
    # else:
    #     pass

    st.markdown('# 🔍 Details') # # 🧐 Details

    st.markdown('# What the app shows')
    write_text_from_file(f'{p}3_Notes.txt',
                        head_lines_to_skip=2)


    # st.image(f'{p}images/whole_app_screenshot.png')
    st.image(f'{p}images/app_overview_models.png')
    st.caption('A cartoon to show which parts of the app\'s demo page are created by each model.')

    st.markdown('## 🏥 Stroke teams')
    write_text_from_file(f'{p}3_Details_Stroke_teams.txt',
                        head_lines_to_skip=2)


    st.markdown('## 📈 Patient data')
    write_text_from_file(f'{p}3_Details_Patient_data.txt',
                        head_lines_to_skip=2)

    st.markdown('## 🔮 Prediction model')
    write_text_from_file(f'{p}3_Details_Prediction_model.txt',
                        head_lines_to_skip=2)


    st.markdown('## ⚖️ SHAP values')


    container_input_highlighted_teams = st.container()
    
    # User inputs
    with st.sidebar:
        st.markdown(
            '## Patient details',
            help=(
                f'''
                🔍 - [Which patient details are included?]({path_to_details}which-features-are-used)

                🔍 - [Why do we model only ten features?]({path_to_details}why-these-features)
                '''
                )
            )
        # Put all of the user input widgets in here later:
        container_input_patient_details = st.container()

        # Add an option for removing plotly_events()
        # which doesn't play well on skinny screens / touch devices.

        st.markdown('-'*50)
        st.markdown('## Advanced options')
        if st.checkbox('Disable interactive plots'):
            use_plotly_events = False
        else:
            use_plotly_events = True
        st.caption(''.join([
            'The clickable plots sometimes appear strange ',
            'on small screens and touch devices, ',
            'so select this option to convert them to normal plots.'
        ]))

    from utilities_ml.inputs import setup_for_app
    from utilities_ml.fixed_params import n_benchmark_teams, \
        default_highlighted_team, display_name_of_default_highlighted_team, \
        explainer_file, starting_probabilities
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
        hb_teams_input
    ) = setup_for_app(
        container_input_highlighted_teams,
        container_input_patient_details,
        )

    # Load in the model:
    from utilities_ml.fixed_params import ml_model_file
    model = utilities_ml.inputs.load_pretrained_model(ml_model_file)
    # Main useful array:
    sorted_results = utilities_ml.main_calculations.\
        predict_treatment(X, model, stroke_teams_list,
                          highlighted_teams_list, benchmark_rank_list,
                          hb_teams_list)
    
    container_shapley_probs = st.container()
    with container_shapley_probs:
        # st.markdown('## ')  # Breathing room
        st.markdown('-' * 50)
        st.info(''.join([
            'The patient details can be viewed and changed ',
            'in the left sidebar.'
            ]), icon='ℹ️')

        container_shap_explanation = st.container()
        with container_shap_explanation:
            # Keep this explanation outside of a .txt file so that the
            # format string for starting probability can be added.
            st.markdown('## :scales: How does the model work?')
            st.markdown(''.join([
                'Here we have visualised the process with waterfall plots.'
            ]))

        # Set up tabs:
        tabs_waterfall = st.tabs([
            'Highlighted teams',
            'All teams',
            'Shifts for highlighted teams',
            'Max/min/median teams',
            ])

        # Use this string in the first two tabs:
        waterfall_explanation_str = ''.join([
                'The features are ordered from largest negative effect on ',
                'probability to largest positive effect. ',
                'The 9 largest features are shown individually and the rest ',
                'are condensed into the "other features" bar. ',
                'This bar mostly contains the effect of the patient _not_ ',
                'attending the other stroke teams.'
            ])

        with tabs_waterfall[1]:
            # Combo waterfall explanation:
            st.markdown(''.join([
                'The following chart shows the waterfall charts for all ',
                'teams. Instead of red and blue bars, each team has ',
                'a series of scatter points connected by lines. ',
                'The features are ordered with the most agreed on features ',
                'at the top, and the ones with more variation lower down. '
            ]), help=f'''
🔍 - [How do I read this plot?]({path_to_details}how-do-you-read-a-wizard-s-hat-plot)
''')

        with tabs_waterfall[2]:
            # Box plots explanation:
            st.markdown('''
                Here the range of shifts in probability caused by
                each feature are shown as violin plots.
                The highlighted teams are overlaid as scatter circles.
                ''', help=f'''
🔍 - [How do I read a violin plot?]({path_to_details}how-do-you-read-a-violin-plot)
                ''')

    # explainer = utilities_ml.inputs.load_explainer()
    explainer_probability = utilities_ml.inputs.load_explainer_probability(
        explainer_file
        )

    # This is down here so that the starting probability is ok.
    with container_shap_explanation:
        st.markdown(
            '''
            + Before the model looks at any of the patient\'s details,
            the patient starts with a base probability''' +
            f' of {100.0*starting_probabilities:.2f}%.',
            help=f'''
🔍 - [Why this base probability?]({path_to_details}why-this-base-probability)
''')
        st.markdown(
            '''

            + The model then looks at the value of each feature of
            the patient in turn, and adjusts this probability upwards
            or downwards. The size of the adjustment depends on which
            stroke team the patient is being assessed by.

            + The final probability is found when all of the features
            have been considered.
            ''')


    # ----- Shapley probabilities -----
    # Make Shapley values for all teams:
    shap_values_probability_extended_all, shap_values_probability_all = \
        utilities_ml.main_calculations.find_shapley_values(
            explainer_probability, X)

    # Make separate arrays of the Shapley values for certain teams.
    # Get indices of highest, most average, and lowest probability teams.
    index_high = sorted_results.iloc[0]['Index']
    index_mid = sorted_results.iloc[int(len(sorted_results)/2)]['Index']
    index_low = sorted_results.iloc[-1]['Index']
    indices_high_mid_low = [index_high, index_mid, index_low]
    # Get indices of highlighted teams:
    indices_highlighted = []
    for team in hb_teams_input:
        if plain_str not in team and bench_str not in team:
            # If it's not the default benchmark or non-benchmark
            # team label, then add this index to the list:
            ind_team = sorted_results['Index'][
                sorted_results['HB team'] == team].values[0]
            indices_highlighted.append(ind_team)

    # Shapley values for the high/mid/low indices:
    shap_values_probability_extended_high_mid_low = \
        shap_values_probability_extended_all[indices_high_mid_low]
    # Shapley values for the highlighted indices:
    shap_values_probability_extended_highlighted = \
        shap_values_probability_extended_all[indices_highlighted]

    # ----- Other grids and dataframes for Shap probabilities:
    # Get big SHAP probability grid:
    grid, grid_cat_sorted, stroke_team_2d, headers = \
        utilities_ml.main_calculations.make_heat_grids(
            headers_X,
            sorted_results['Stroke team'],
            sorted_results['Index'],
            shap_values_probability_all
            )
    # These grids have teams in the same order as sorted_results.

    # Pick out the subset of benchmark teams:
    inds_bench = np.where(
        sorted_results['Benchmark rank'].to_numpy() <= n_benchmark_teams)[0]
    inds_nonbench = np.where(
        sorted_results['Benchmark rank'].to_numpy() > n_benchmark_teams)[0]
    # Make separate grids of just the benchmark or non-benchmark teams:
    grid_cat_bench = grid_cat_sorted[:, inds_bench]
    grid_cat_nonbench = grid_cat_sorted[:, inds_nonbench]

    # Make a list of the input patient data for labelling
    # features+values on e.g. the combined waterfall plot.
    # Pull out the feature values:
    patient_data_waterfall = X.iloc[0][:9].to_numpy()
    # Add empty value for stroke team attended:
    patient_data_waterfall = np.append(patient_data_waterfall, '')
    # Find which values are 0/1 choice and can be changed to no/yes:
    if 'SAMueL-1' in model_version:
        features_yn = [
            'Infarction',
            'Precise onset time',
            'Use of AF anticoagulants',
            'Onset during sleep',
        ]
    else:
        features_yn = [
            'infarction',
            'precise onset known',
            'use of AF anticoagulants',
            'onset during sleep'
        ]
    for feature in features_yn:
        i = np.where(np.array(headers_X) == feature)[0]
        # Annoying nested list to pacify DeprecationWarning for
        # checking for element of empty array.
        if patient_data_waterfall[i].size > 0:
            if patient_data_waterfall[i] > 0:
                patient_data_waterfall[i] = 'Yes'
            else:
                patient_data_waterfall[i] = 'No'
        else:
            patient_data_waterfall[i] = 'No'
    # Resulting list format e.g.:
    #     [15, 'Yes', 15, 'Yes', 0, 'No', 90, 'No', 72.5, '']
    # where headers_X provides the feature names to match the values.

    # Make dataframe for combo waterfalls:
    # patient_data_waterfall is returned here with the order of the
    # values switched to match the order the features are plotted in
    # in the combo waterfall.
    df_waterfalls, final_probs, patient_data_waterfall = \
        utilities_ml.main_calculations.make_waterfall_df(
            grid_cat_sorted,
            headers,
            sorted_results['Stroke team'],
            sorted_results['Highlighted team'],
            sorted_results['HB team'],
            patient_data_waterfall,
            base_values=starting_probabilities
            )
    # Columns of df_waterfalls:
    #     Sorted rank
    #     Stroke team
    #     Probabilities
    #     Prob shift
    #     Prob final
    #     Features
    #     Highlighted team
    #     HB team

    with container_shapley_probs:

        with tabs_waterfall[0]:
            # Individual waterfalls for highlighted teams.
            if len(indices_highlighted) < 1:
                # Nothing to see here
                st.write('No teams are highlighted.')
            else:
                # Individual waterfalls for the highlighted teams.
                st.markdown(waterfall_explanation_str, help=f'''
🔍 - [How do I read a waterfall plot?]({path_to_details}how-do-you-read-a-waterfall-plot)
                ''')
                draw_sneaky_bar()
                utilities_ml.container_waterfalls.show_waterfalls_highlighted(
                    shap_values_probability_extended_highlighted,
                    indices_highlighted,
                    sorted_results,
                    default_highlighted_team,
                    display_name_of_default_highlighted_team,
                    model_version
                    )

        with tabs_waterfall[1]:
            # Combo waterfall (all teams).
            draw_sneaky_bar()
            utilities_ml.container_combo_waterfall.plot_combo_waterfalls(
                df_waterfalls,
                sorted_results,
                final_probs,
                patient_data_waterfall,
                default_highlighted_team,
                display_name_of_default_highlighted_team,
                use_plotly_events
                )
            st.caption('''
                If no graph appears above this message,
                try changing the patient details
                to reload the app.
                ''')

        with tabs_waterfall[2]:
            # Box plots:
            utilities_ml.container_combo_waterfall.box_plot_of_prob_shifts(
                grid_cat_sorted,
                grid_cat_bench,
                grid_cat_nonbench,
                headers,
                sorted_results,
                hb_teams_input,
                default_highlighted_team,
                display_name_of_default_highlighted_team,
                starting_probabilities
                )

        with tabs_waterfall[3]:
            # Individual waterfalls for the teams with the
            # max / median / min probabilities of thrombolysis.
            st.markdown(waterfall_explanation_str, help=f'''
🔍 - [How do I read a waterfall plot?]({path_to_details}how-do-you-read-a-waterfall-plot)
                ''')
            draw_sneaky_bar()
            utilities_ml.container_waterfalls.show_waterfalls_max_med_min(
                shap_values_probability_extended_high_mid_low,
                indices_high_mid_low,
                sorted_results,
                default_highlighted_team,
                display_name_of_default_highlighted_team,
                model_version
                )
            
    write_text_from_file(f'{p}3_Details_SHAP.txt',
                        head_lines_to_skip=2)

    feats_dict = {
        'Anticoagulants':'afib_anticoagulant',
        'Age':'age',
        'Arrival to scan time':'arrival_to_scan_time',
        'Infarction':'infarction',
        'Onset during sleep':'onset_during_sleep',
        'Onset to arrival time':'onset_to_arrival_time',
        'Precise onset known':'precise_onset_known',
        'Prior disability':'prior_disability',
        'Stroke severity':'stroke_severity'
    }
    feat_pretty = st.selectbox(
        'Select a feature to display', 
        options=list(feats_dict.keys())
        )
    feat = feats_dict[feat_pretty]
    image_name = f'04_xgb_10_features_thrombolysis_shap_boxplot_{feat}.jpg'
    st.image(f'{p}images/{image_name}')

    # Write some information about that image.
    # First define a string with the info for each image,
    # and then use another dictionary to pick out the relevant string.
    afib_anticoagulant_str = '''
When the patient is on anticoagulants, the "feature value" at the bottom is 1.
The average SHAP value is below -1.5 when the patient is on anticoagulants.
This means that the patient is considerably less likely to be treated
than if they were not on blood-thinning medication.
'''
    age_str = '''
The average SHAP value decreases as age increases.
For patients below the age of 80, the SHAP value is positive and so
there is a small increase in the chance of thrombolysis.
For patients above the age of 80, the SHAP value is negative and becomes
more and more negative with increasing age. This reduces the chance of these
older patients receiving thrombolysis.
'''
    arrival_to_scan_time_str = '''
The average SHAP value decreases as the time between arrival at hospital
and receiving a brain scan increases. 
On average, every extra 30 minutes decreases the SHAP value by around 0.5.
That means that after every hour on average, the chance of receiving
thrombolysis becomes three times smaller.
'''
    infarction_str = '''
When the "feature value" at the bottom is 1, the patient has an infarction
which is also called a blood clot. When the patient does not have a blood clot,
the average SHAP value is very negative at around minus 9. This means that
the chance of being given thrombolysis becomes very small. This reflects how
in real life, clot-busting drugs should not be given when there is no clot.
'''
    onset_during_sleep_str = '''
When the "feature value" at the bottom is 1, the patient\'s stroke began
while they were asleep. This means that the exact time that the stroke
started is less accurately known, and so the chance of their being treated
with thrombolysis is reduced considerably. The average SHAP value in this
case is below minus 1.
'''
    onset_to_arrival_time_str = '''
The average SHAP value decreases as the time between the start of the stroke and
the arrival at hospital increases. Patients arriving before 120 minutes are
slightly more likely to receive treatment. Patients arriving after 120 minutes
are slightly less likely to receive treatment.
'''
    precise_onset_known_str = '''
When the "feature value" at the bottom is 0, the time that the stroke started
is not known precisely. The average SHAP value in this case is almost minus 1.
This makes the patient somewhat less likely to receive thrombolysis.
'''
    prior_disability_str = '''
As the prior disability score increases, the average SHAP value decreases.
This means that patients who had a higher level of disability before the
stroke began are somewhat less likely to be treated than patients with
no pre-stroke disability.
'''
    stroke_severity_str = '''
In this chart, each separate score for stroke severity has its own box.
The average SHAP value is negative for low scores below 5
and for high scores above 32. For scores between those values,
the average SHAP value is positive. This means that the most mild and the
most severe strokes are somewhat less likely to be treated with thrombolysis,
and the most moderate strokes are somewhat more likely to be treated 
with thrombolysis.
'''

    feats_str_dict = {
        'Anticoagulants':afib_anticoagulant_str,
        'Age':age_str,
        'Arrival to scan time':arrival_to_scan_time_str,
        'Infarction':infarction_str,
        'Onset during sleep':onset_during_sleep_str,
        'Onset to arrival time':onset_to_arrival_time_str,
        'Precise onset known':precise_onset_known_str,
        'Prior disability':prior_disability_str,
        'Stroke severity':stroke_severity_str
    }
    st.markdown(feats_str_dict[feat_pretty])

    st.markdown('''## ⛲ Waterfall plots

### How do you read a waterfall plot?

The waterfall plot has a number of rows. Each row is labelled with a patient detail name, and each row contains a coloured block. The blocks show the effect of that patient detail on the overall probability of receiving thrombolysis. Larger blocks mean a larger effect. The size of the effect is also written next to each block.

Always start at the top at the place marked "start probability". Then move downwards row by row, adding or removing probability, until you reach the bottom of the plot. The "end probability" that you are left with after adding on the effect of all of the coloured blocks is the overall probability of thrombolysis for this patient.

The features are ordered from largest negative effect on probability to largest positive effect. The 9 largest features are shown individually and the rest are condensed into the "other features" bar. This bar mostly contains the effect of the patient _not_ attending the other stroke teams.
    ''')
    st.image(f'{p}images/waterfall.png')
    st.caption('An example waterfall plot.')

    st.markdown('''
### What does this example show?

The patient starts with a probability of treatment of 34.82%. Then working downwards, the following changes happen:
                ''')
    with st.expander('Detailed look'):
        write_text_from_file(f'{p}3_Details_Waterfall.txt',
                            head_lines_to_skip=2)

    st.markdown('''## 🧙‍♀️ Wizard hat plots
            
The wizard hat plot is so named because the resulting lines look like the shape of a wonky crooked cartoon wizard's hat.
            
The plot combines the data in all of the separate stroke teams' SHAP waterfall plots.
            
### How do you read a wizard's hat plot?

This plot should be read in the same way as a waterfall plot. Start at the top and work downwards row by row, shifting the "probability so far" in each row left or right a bit depending on the patient details.
            
Instead of drawing coloured blocks to show the shifts in probability, this graph simply connects the points with a straight line. Each stroke team gets one coloured line that runs from the top row to the bottom row.
            
There are so many lines that it is difficult to follow the progress of individual teams unless they are highlighted in a different colour. This means that the graph is best used to get an idea of how the patient details affect the probability of treatment for the different stroke teams.
            
The features are ordered with the most agreed on features at the top, and the ones with more variation lower down.
                ''')
    st.image(f'{p}images/wizard.png')
    st.caption('An example wizard\'s hat plot.')
    write_text_from_file(f'{p}3_Details_Wizard_Hat.txt',
                        head_lines_to_skip=2)

    st.markdown('## 🎻 Violin plots')
    st.markdown('''
The violin plot shows the distribution of values across all stroke teams. These values are usually the probabilities of receiving thrombolysis.

### How do you read a violin plot?

The range of possible values is shown from left to right. The violin is fatter where there are more stroke teams with that value. Usually there are one or two values that most of the teams are close to and so the violin will appear fattest there. 

The black lines show the smallest, average (median) and largest probabilities out of all of the teams.

                ''')
    cols_violin = st.columns(2)
    with cols_violin[0]:
        st.image(f'{p}images/violin.png')
        st.caption(''.join([
            'Three example violin plots. ',
            'The top row shows the range of values for all teams. ',
            'The middle row shows only benchmark teams, and ',
            'the bottom row shows only non-benchmark teams. ',
            'The red circle highlights the value of one particular team.'
            ]))
    with cols_violin[1]:
        st.markdown('''
### What does this example show?

The top row shows that most of the teams have a probability above 80%. The violin is fattest around 85% to 95% and so most of the values lie in that range. 

We also have separate rows for only the benchmark and only the non-benchmark teams. There are 25 benchmark teams and 94 non-benchmark teams. Since most of the "all teams" row is made up of non-benchmark teams, the non-benchmark teams violin is usually similar in shape to the all team violin.

The middle row shows only the benchmark teams. The probabilities here are more squashed up towards the right-hand-side, with most lying between 90% to 100%. This means that generally the benchmark teams result in higher probabilities than you generally see in all of the teams or in only the non-benchmark teams.
                    ''')
    st.markdown('''
            

### What do the circle markers mean?

For the plots like in this example, the circle markers on top of the violins mark the values for any teams that have been highlighted. 

There is no meaning behind how high up or low down the circle markers appear. They are moved up or down to reduce the overlap between circles with a similar place on the graph.
    '''
                )

    st.markdown('-'*50)
    st.markdown('# How the app works')


    st.markdown('## 🧮 The data')
    st.markdown('''
### What data do we use?

The original data set comes from the Sentinel Stroke National Audit Programme (SSNAP). The data has the details of emergency hospital admissions for several hundreds of thousands of patients with stroke between the years 2016 and 2021. The data contains information such as the time between the start of the stroke and the arrival at hospital, what type of stroke it was, and which treatments were given.

The models only use the ten features that are shown here in the app.

💫 _More detail:_
+ Technical notes on the data set: [here](https://samuel-book.github.io/samuel-1/introduction/data.html).
    ''')
    # cols_data = st.columns(2)
    # with cols_data[0]:
    st.markdown('''
    ### What data goes into the models?''')
    st.image(f'{p}images/process_data_split.png')
    # with cols_data[1]:
    st.markdown('''
We use a subset of the full data set to create the models.

The decision model is created using a set of training data with around 100,000 patients.
This includes only patients whose onset to scan time is four hours or less.
We then use a testing data set of 10,000 patients to test the accuracy of the model.
    '''
        )

    st.markdown('''## 🔮 Prediction model
For the simplest explanation of what the app does:
1. The input patient details are converted into a table that the model knows how to read.
2. The details are passed to the decision model.
3. The decision model calculates the probability of thrombolysis for each patient in the table.
4. The probabilities are also converted to a simple "yes" or "no" for whether a patient would have received thrombolysis. We set any patient with at least 50% probability as "yes".
    ''')

    st.image(f'{p}images/process_decision_model.png')


    st.markdown('''
### Creating the decision model
            
💫 _More detail:_
+ A plain English summary of machine learning models: [here](https://samuel-book.github.io/samuel-2/samuel_shap_paper_1/introduction/intro.html)

The decision model is a type of machine learning model called an XGBoost Classifier.
We start off with a base model that has never been shown any data but is ready to learn.
Then we give this model the training data set.
The model goes through each patient in turn and decides whether or not they should receive thrombolysis.

At first the model does an awful job because it doesn't have much experience.
As the model sees more and more patients, it learns more about the general patterns that affect how likely patients are to be treated.
When the model has seen all of the patients in the training data set, it has become quite accurate.
Whether it says the patient would receive thrombolysis usually matches the actual case in real life.
    ''')
    st.image(f'{p}images/process_create_decision_model.png')


    st.markdown('''

### How do we know it works?

We use a set of testing data from the real-life patient data set. For this data we know whether each patient received thrombolysis in real life. We can pass that data to the decision model to predict whether each patient was thrombolysed, and then see how often the prediction matched reality.

(To do - write about the more robust checks including ROC-AUC.)
    ''')

    st.markdown('''
### Creating the SHAP explainer model

The SHAP explainer model is created directly from the decision model. 

We actually create two SHAP explainers. 
+ One uses only the decision model and works in units of log-odds. These units are technically more useful than simple probabilities, but they are also more difficult to understand intuitively. This model is used to define the benchmark teams.
+ The second SHAP explainer works in units of probability, and also uses the testing data set to calibrate the probabilities. This model is used to show the decision model's workings in this app.
    ''')


    st.image(f'{p}images/process_create_shap_models.png')

    st.markdown('''
## ⚖️ SHAP values  

For the simplest explanation of what the app does:
1. The input patient details are converted into a table that the model knows how to read. This table can also be read by the SHAP explainer model.
2. The data is passed to the SHAP explainer model.
3. The model provides a series of SHAP values. The values for each patient are not affected by any other patients. A SHAP value is calculated for each input patient detail and for each stroke team.
    ''')

    st.image(f'{p}images/process_shap_model.png')

    st.markdown('''### Creating a waterfall plot
            
To create a waterfall plot:
1. Separate off one patient's data.
2. Sort the SHAP values from smallest to biggest.
3. Add up all of the 119 values in the middle. Each of these should have a small effect compared with the remaining 9 values.
4. Plot the base probability.
5. For each value in the table in turn, add it to the probability so far. Work downwards along the waterfall until the plot is complete. ''')
    st.image(f'{p}images/process_waterfall.png')

    st.markdown('''### Creating a wizard\'s hat plot
            
To create a wizard's hat plot:
1. Combine all of the separate teams' values into one value by adding them together.
2. For each feature, calculate the standard deviation across all of its SHAP values. This is a measure of how similar the values are.
3. Sort the table from the smallest standard deviation (most similar values) to the largest. This is the order in which we'll plot the data.
4. For one team at a time, go back to the original table and:
    1. Draw the base probability.
    2. Calculate how much the first feature in the list changes the base probability, and plot that on the next row.
    3. Continue working down the waterfall until you reach the bottom.
5. In the final row, take all of the final probabilities of thrombolysis and create a violin plot to show the distribution of values.
    ''')
    st.image(f'{p}images/process_combo_waterfall.png')

                
    st.markdown('## 🎯 Benchmark teams')
    st.markdown('''
### What are benchmark teams?

The benchmark teams are consistently more likely than average to choose to thrombolyse any patient. 

### How are the benchmark teams picked?
    ''')
    cols_bench = st.columns(2)
    with cols_bench[0]:
        st.image(f'{p}images/flowchart_define_benchmarks.png')
    with cols_bench[1]:
        write_text_from_file(f'{p}3_Details_Benchmark_teams.txt',
                            head_lines_to_skip=2)

if __name__ == '__main__':
    main()