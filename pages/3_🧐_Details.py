import streamlit as st

from utilities_ml.fixed_params import page_setup
from utilities_ml.inputs import write_text_from_file

page_setup()

p = 'pages/text_for_pages/'

write_text_from_file(f'{p}3_Notes.txt',
                     head_lines_to_skip=2)

st.markdown('## 🧮 Original data')
cols_data = st.columns(2)
with cols_data[0]:
    st.image(f'{p}images/flowchart_split_data.png')
with cols_data[1]:
    st.markdown('''
### What data was the model trained on?

We use a set of training data from the real-life patient data set. This includes only patients whose onset to scan time is four hours or less. 
There are around 100,000 patients in the training data set
and 10,000 patients in the testing data set.
''')

st.markdown('## 🏥 Stroke teams')
write_text_from_file(f'{p}3_Details_Stroke_teams.txt',
                     head_lines_to_skip=2)


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

st.markdown('## 🙂 Patient data')
write_text_from_file(f'{p}3_Details_Patient_data.txt',
                     head_lines_to_skip=2)

st.markdown('## 🔮 Prediction model')
cols_model = st.columns(2)
with cols_model[0]:
    st.image(f'{p}images/flowchart_make_models.png')
with cols_model[1]:
    write_text_from_file(f'{p}3_Details_Prediction_model.txt',
                        head_lines_to_skip=2)


st.markdown('## ⚖️ SHAP values')
write_text_from_file(f'{p}3_Details_SHAP.txt',
                     head_lines_to_skip=2)

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