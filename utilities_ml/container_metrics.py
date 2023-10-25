import streamlit as st


def main(sorted_results, n_benchmark_teams):
    # Show metrics for all, benchmark, non-bench teams.

    # Benchmark teams:
    # sorted_results['Benchmark rank']

    inds_benchmark = sorted_results['Benchmark rank'] <= n_benchmark_teams

    results_all = sorted_results
    results_benchmark = sorted_results.loc[inds_benchmark]
    results_non_benchmark = sorted_results.loc[~inds_benchmark]

    # Number of entries that would thrombolyse:
    n_thrombolyse_all = results_all.Thrombolyse.sum()
    n_thrombolyse_benchmark = results_benchmark.Thrombolyse.sum()
    n_thrombolyse_non_benchmark = results_non_benchmark.Thrombolyse.sum()

    # Total number of entries:
    n_all = len(results_all)
    n_benchmark = len(results_benchmark)
    n_non_benchmark = len(results_non_benchmark)

    # Percentage of entries that would thrombolyse:
    perc_thrombolyse_all = 100.0 * n_thrombolyse_all / n_all
    perc_thrombolyse_benchmark = 100.0 * n_thrombolyse_benchmark / n_benchmark
    perc_thrombolyse_non_benchmark = (
        100.0 * n_thrombolyse_non_benchmark / n_non_benchmark)

    cols_markers = st.columns(3)
    with cols_markers[0]:
        st.markdown('Benchmark teams:')
        st.markdown(make_marker_grid(
            n_thrombolyse_benchmark,
            n_benchmark - n_thrombolyse_benchmark)
            )

    with cols_markers[1]:
        st.markdown('Non-benchmark teams:')
        st.markdown(make_marker_grid(
            n_thrombolyse_non_benchmark,
            n_non_benchmark - n_thrombolyse_non_benchmark)
            )


    # cols = st.columns(4, gap='large')
    with cols_markers[2]:
        yes_str = (
            ':heavy_check_mark:' +
            f' {n_thrombolyse_all} team' +
            ('s' if n_thrombolyse_all != 1 else '')
            )
        no_str = (
            ':x:' +
            f' {n_all - n_thrombolyse_all} team' +
            ('s' if n_all - n_thrombolyse_all != 1
             else '')
        )
        st.markdown(
            f'''
            All teams  
            {yes_str}  
            {no_str}
            '''
        )

        yes_str = (
            ':heavy_check_mark:' +
            f' {n_thrombolyse_benchmark} team' +
            ('s' if n_thrombolyse_benchmark != 1 else '')
            )
        no_str = (
            ':x: ' +
            f'{n_benchmark - n_thrombolyse_benchmark} team' +
            ('s' if n_benchmark - n_thrombolyse_benchmark != 1
             else '')
            )
        st.markdown(
            f'''
            Benchmark teams  
            {yes_str}  
            {no_str}
            '''
        )

        yes_str = (
            ':heavy_check_mark:' +
            f' {n_thrombolyse_non_benchmark} team' +
            ('s' if n_thrombolyse_non_benchmark != 1 else '')
        )
        no_str = (
            ':x:' +
            f' {n_non_benchmark - n_thrombolyse_non_benchmark} team' +
            ('s' if n_non_benchmark - n_thrombolyse_non_benchmark != 1
             else '')
        )
        st.markdown(
            f'''
            Non-benchmark teams  
            {yes_str}  
            {no_str}
            '''
        )

    # Write benchmark decision:
    extra_str = (' would' if perc_thrombolyse_benchmark >= 50.0
                 else ' would not')
    decision_emoji = (':heavy_check_mark:'
                        if perc_thrombolyse_benchmark >= 50.0 else ':x:')
    st.error(
        f'''
        __Benchmark decision:__ {decision_emoji}{extra_str} thrombolyse
        '''  # this patient.'
    )


def make_marker_grid(n_yes, n_no):
    # Format the string so ten markers appear per row.
    count = n_yes
    full_str = ''
    ticks = True
    while ticks:
        if count >= 10:
            full_str += ':heavy_check_mark:' * 10
            full_str += '''  
                        '''
            count -= 10
        else:
            full_str += ':heavy_check_mark:' * count
            ticks = False
    count_non = n_no
    left_in_row = 10 - count
    while count_non > 0:
        if count_non >= left_in_row:
            full_str += ':x:' * left_in_row
            full_str += '''  
                        '''
            count_non -= left_in_row
            left_in_row = 10
        else:
            full_str += ':x:' * count_non
            count_non = 0
    return full_str
