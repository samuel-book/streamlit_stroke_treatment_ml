import streamlit as st


def main(sorted_results):
    # Show metrics for all, benchmark, non-bench teams.
    
    # Benchmark teams:
    # sorted_results['Benchmark rank']

    inds_benchmark = sorted_results['Benchmark rank'] <= 30

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
    
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            f'All {n_all} teams',
            f'{perc_thrombolyse_all:.0f}%'
            )
        st.write(''.join([
            ':heavy_check_mark:' + f' {n_thrombolyse_all} teams '
            ':x:' + f' {n_all - n_thrombolyse_all} teams'
            ]))

    with cols[1]:
        st.metric(
            f'{n_benchmark} Benchmark teams',
            f'{perc_thrombolyse_benchmark:.0f}%'
            )
        st.write(''.join([
            ':heavy_check_mark:' + f' {n_thrombolyse_benchmark} teams '
            ':x:' + f' {n_benchmark - n_thrombolyse_benchmark} teams'
            ]))


    with cols[2]:
        st.metric(
            f'{n_non_benchmark} Non-benchmark teams',
            f'{perc_thrombolyse_non_benchmark:.0f}%'
            )
        st.write(''.join([
            ':heavy_check_mark:' + f' {n_thrombolyse_non_benchmark} teams '
            ':x:' + f' {n_non_benchmark - n_thrombolyse_non_benchmark} teams'
            ]))


    # Write benchmark decision:
    extra_str = '' if perc_thrombolyse_benchmark >= 50.0 else ' do not'
    decision_emoji = ':heavy_check_mark:' if perc_thrombolyse_benchmark >= 50.0 else ':x:'
    st.markdown(''.join([
        '__Benchmark decision:__ ',
        decision_emoji,
        extra_str,
        ' thrombolyse this patient.'
        ]))

