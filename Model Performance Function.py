import pandas as pd
import numpy as np


def model_performance(df, amt, pred_score, chargeback, current_reject, n_samples_vec, reject_multiplier=0.5, chargeback_multiplier=2, two_models=True, current_decline_precision=0.5):
    """Summary

    Args:
        df (pandas data frame): input data with labels and prediction score.
        amt (str): column name containing dollar amount of orders.
        pred_score (str): column name containing the ML score.
        chargeback (str): column name containing chargeback field (chargeback = 1, non chargeback = 0).
        current_reject (str): old reject label, (current_reject = 1, non current_reject = 0).
        n_samples_vec (list): a list, indicating the number of tranactions get declined in the input dataset.
        reject_multiplier (float, optional): reject multiplier, reject multiplier = (amount of revenue loss)/(one dollar of transaction declined).
        chargeback_multiplier (float, optional): chargeback multiplier, chargeback multiplier = (amount of revenue loss)/(one dollar of chargeback)
        two_models (bool, optional): True (two model performance comparisons), False (one model performance).
        current_decline_precision (float, optional): assumed client decline precision, only when two_models = True.

    Returns:
        TYPE: pandas data frame with performance evaluation results
        current_decline_precision: current system decline precision
        total_count: number of orders in the input data
        total_amt: total amount of orders in the input data
        total_cb_count: number of chargebacks in the input data
        total_cb_amt: chargeback amount of all the orders in the input data
        min_score_decline: minimum model score of declined transactions in the new ML model
        count_decline: number of orders declined using min_score_decline as the threshold
        amt_decline: amount of orders declined using min_score_decline as the threshold
        prop_data_decline: proportion of orders (count based) declined using min_score_decline as the threshold
        prop_amt_decline: proportion of orders (amount based) declined using min_score_decline as the threshold
        count_precision_current: count based precision of current system
        count_recall_current: count based recall of current system
        count_precision_new: count based precision using min_score_decline as the threshold
        count_recall_new: count based recall using min_score_decline as the threshold
        amt_precision_current: amount based precision of current system
        amt_recall_current: amount based recall of current system
        amt_precision_new: amount based precision using min_score_decline as the threshold
        amt_recall_new: amount based recall using min_score_decline as the threshold
        amt_cb_new_decline_catch: amount of chargeback caught using min_score_decline as the threshold
        amt_cb_current_decline_catch: amount of chargeback caught of current system
        count_new_cb: number of chargeback using min_score_decline as the threshold
        count_new_reject: number of reject using min_score_decline as the threshold
        count_new_accept_noncb: number of good accepted transactions using min_score_decline as the threshold
        amt_new_cb: amount of chargeback using min_score_decline as the threshold
        amt_new_reject: amount of reject using min_score_decline as the threshold
        amt_new_accept_noncb: amount of good accepted transactions using min_score_decline as the threshold
        count_current_cb: number of chargeback of current system
        count_current_reject: number of reject of current system
        count_current_accept_noncb: number of good accepted transactions of current system
        amt_current_cb: amount of chargeback of current system
        amt_current_reject: amount of reject of current system
        amt_current_accept_noncb: amount of good accepted transactions of current system
        amt_cb_reduction_wo_mp: amount of chargeback reduction without any multiplier
        tpv_enablement_wo_mp: tpv enablement without any multiplier
        net_revenue: sum of amt_cb_reduction_wo_mp and tpv_enablement_wo_mp
        tpv_enablement_w_mp: tpv enablement with the reject_multiplier and chargeback_multiplier
        cb_caught_efficiency: amount of true positive/amount of false positive using min_score_decline as the threshold, the higher score means higher cb_caught_efficiency efficiency
        amt_fp_current_tn_new: amount of tranactions which are false positive in current system and true negative using min_score_decline as the threshold
        amt_fp_current_fp_new: amount of tranactions which are false positive in current system and false positive using min_score_decline as the threshold
        amt_tp_current_fn_new: amount of tranactions which are true positive in current system and false negative using min_score_decline as the threshold
        amt_tp_current_tp_new: amount of tranactions which are true positive in current system and true positive using min_score_decline as the threshold
        amt_tn_current_tn_new: amount of tranactions which are true negative in current system and true negative using min_score_decline as the threshold
        amt_tn_current_fp_new: amount of tranactions which are true negative in current system and false negative using min_score_decline as the threshold
        amt_fn_current_fn_new: amount of tranactions which are false negative in current system and false negative using min_score_decline as the threshold
        amt_fn_current_tp_new: amount of tranactions which are false negative in current system and true positive using min_score_decline as the threshold
        count_fp_current_tn_new: count of tranactions which are false positive in current system and true negative using min_score_decline as the threshold
        count_fp_current_fp_new: count of tranactions which are false positive in current system and false positive using min_score_decline as the threshold
        count_tp_current_fn_new: count of tranactions which are true positive in current system and false negative using min_score_decline as the threshold
        count_tp_current_tp_new: count of tranactions which are true positive in current system and true positive using min_score_decline as the threshold
        count_tn_current_tn_new: count of tranactions which are true negative in current system and true negative using min_score_decline as the threshold
        count_tn_current_fp_new: count of tranactions which are true negative in current system and false negative using min_score_decline as the threshold
        count_fn_current_fn_new: count of tranactions which are false negative in current system and false negative using min_score_decline as the threshold
        count_fn_current_tp_new: count of tranactions which are false negative in current system and true positive using min_score_decline as the threshold
    """
    df = df.sort_values(by=pred_score, ascending=False)
    df = df.reset_index(drop=True)

    total_count = df.shape[0]
    total_cb_count = df[chargeback].sum()
    total_amt = df[amt].sum()
    total_cb_amt = df.loc[df[chargeback] == 1][amt].sum()

    columns = [
        'current_decline_precision',
        'total_count',
        'total_amt',
        'total_cb_count',
        'total_cb_amt',
        'min_score_decline',
        'count_decline',
        'amt_decline',
        'prop_data_decline',
        'prop_amt_decline',
        'count_precision_current',
        'count_recall_current',
        'count_precision_new',
        'count_recall_new',
        'amt_precision_current',
        'amt_recall_current',
        'amt_precision_new',
        'amt_recall_new',
        'amt_cb_new_decline_catch',
        'amt_cb_current_decline_catch',
        'count_new_cb',
        'count_new_reject',
        'count_new_accept_noncb',
        'amt_new_cb',
        'amt_new_reject',
        'amt_new_accept_noncb',
        'count_current_cb',
        'count_current_reject',
        'count_current_accept_noncb',
        'amt_current_cb',
        'amt_current_reject',
        'amt_current_accept_noncb',
        'amt_cb_reduction_wo_mp',
        'tpv_enablement_wo_mp',
        'net_revenue',
        'tpv_enablement_w_mp',
        'cb_caught_efficiency',
        'amt_fp_current_tn_new',
        'amt_fp_current_fp_new',
        'amt_tp_current_fn_new',
        'amt_tp_current_tp_new',
        'amt_tn_current_tn_new',
        'amt_tn_current_fp_new',
        'amt_fn_current_fn_new',
        'amt_fn_current_tp_new',
        'count_fp_current_tn_new',
        'count_fp_current_fp_new',
        'count_tp_current_fn_new',
        'count_tp_current_tp_new',
        'count_tn_current_tn_new',
        'count_tn_current_fp_new',
        'count_fn_current_fn_new',
        'count_fn_current_tp_new'
    ]

    scores = pd.DataFrame(columns=columns, index=n_samples_vec)
    # scores = pd.DataFrame(columns=columns)

    for n_samples in n_samples_vec:
        df['reject'] = 0
        df.loc[:n_samples, 'reject'] = 1

        min_score_decline = df.loc[df['reject'] == 1][pred_score].iloc[-1]
        count_decline = df['reject'].sum()
        amt_decline = df.loc[df['reject'] == 1][amt].sum()
        prop_data_decline = 100 * count_decline / total_count
        prop_amt_decline = 100 * amt_decline / total_amt

        count_tn_current_tn_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 0) & (df['reject'] == 0)].shape[0]
        count_tn_current_fp_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 0) & (df['reject'] == 1)].shape[0]
        count_fn_current_fn_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 1) & (df['reject'] == 0)].shape[0]
        count_fn_current_tp_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 1) & (df['reject'] == 1)].shape[0]

        amt_tn_current_tn_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 0) & (df['reject'] == 0)][amt].sum()
        amt_tn_current_fp_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 0) & (df['reject'] == 1)][amt].sum()
        amt_fn_current_fn_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 1) & (df['reject'] == 0)][amt].sum()
        amt_fn_current_tp_new = df.loc[(df[current_reject] == 0) & (df[chargeback] == 1) & (df['reject'] == 1)][amt].sum()

        if not two_models:
            current_decline_precision = np.nan
            # Default old declined transactions are count 0 and amt 0 for one model case
            count_fp_current_tn_new, count_fp_current_fp_new = 0, 0
            count_tp_current_fn_new, count_tp_current_tp_new = 0, 0
            amt_fp_current_tn_new, amt_fp_current_fp_new = 0, 0
            amt_tp_current_fn_new, amt_tp_current_tp_new = 0, 0
            # Default old precision and recall for one model case
            count_precision_current, count_recall_current = np.nan, np.nan
            amt_precision_current, amt_recall_current = np.nan, np.nan

        else:
            count_current_declined_new_approve = df.loc[(df[current_reject] == 1) & (df['reject'] == 0)].shape[0]
            count_current_declined_new_decline = df.loc[(df[current_reject] == 1) & (df['reject'] == 1)].shape[0]
            amt_current_declined_new_approve = df.loc[(df[current_reject] == 1) & (df['reject'] == 0)][amt].sum()
            amt_current_declined_new_decline = df.loc[(df[current_reject] == 1) & (df['reject'] == 1)][amt].sum()

            count_current_declined = count_current_declined_new_approve + count_current_declined_new_decline
            amt_current_declined = amt_current_declined_new_approve + amt_current_declined_new_decline

            current_fp_rate = 1 - current_decline_precision
            new_approve_rate = count_current_declined_new_approve / count_current_declined
            new_decline_rate = 1 - new_approve_rate

            count_fp_current_fp_new = count_current_declined * current_fp_rate * new_decline_rate
            count_fp_current_tn_new = count_current_declined * current_fp_rate * new_approve_rate
            count_tp_current_tp_new = count_current_declined * current_decline_precision * new_decline_rate
            count_tp_current_fn_new = count_current_declined * current_decline_precision * new_approve_rate

            amt_fp_current_fp_new = amt_current_declined * current_fp_rate * new_decline_rate
            amt_fp_current_tn_new = amt_current_declined * current_fp_rate * new_approve_rate
            amt_tp_current_tp_new = amt_current_declined * current_decline_precision * new_decline_rate
            amt_tp_current_fn_new = amt_current_declined * current_decline_precision * new_approve_rate

            # Calculate old the precision and recall
            count_current_tp = count_tp_current_fn_new + count_tp_current_tp_new
            count_current_fp = count_fp_current_tn_new + count_fp_current_fp_new
            count_current_fn = count_fn_current_fn_new + count_fn_current_tp_new

            amt_current_tp = amt_tp_current_fn_new + amt_tp_current_tp_new
            amt_current_fp = amt_fp_current_tn_new + amt_fp_current_fp_new
            amt_current_fn = amt_fn_current_fn_new + amt_fn_current_tp_new

            count_precision_current = 100 * count_current_tp / (count_current_tp + count_current_fp)
            count_recall_current = 100 * count_current_tp / (count_current_tp + count_current_fn)
            amt_precision_current = 100 * amt_current_tp / (amt_current_tp + amt_current_fp)
            amt_recall_current = 100 * amt_current_tp / (amt_current_tp + amt_current_fn)

        # Calculate new the precision and recall
        count_new_tp = count_tp_current_tp_new + count_fn_current_tp_new
        count_new_fp = count_fp_current_fp_new + count_tn_current_fp_new
        count_new_fn = count_tp_current_fn_new + count_fn_current_fn_new

        amt_new_tp = amt_tp_current_tp_new + amt_fn_current_tp_new
        amt_new_fp = amt_fp_current_fp_new + amt_tn_current_fp_new
        amt_new_fn = amt_tp_current_fn_new + amt_fn_current_fn_new

        count_new_positive = count_new_tp + count_new_fp
        count_precision_new = np.nan
        if count_new_positive:
            count_precision_new = 100 * count_new_tp / count_new_positive

        count_total_positive = count_new_tp + count_new_fn
        count_recall_new = np.nan
        if count_total_positive:
            count_recall_new = 100 * count_new_tp / count_total_positive

        amt_new_positive = amt_new_tp + amt_new_fp
        amt_precision_new = np.nan
        if amt_new_positive:
            amt_precision_new = 100 * amt_new_tp / amt_new_positive

        amt_total_positive = amt_new_tp + amt_new_fn
        amt_recall_new = np.nan
        if amt_total_positive:
            amt_recall_new = 100 * amt_new_tp / amt_total_positive

        # Calculate the Revenue Metrics: Without Fraud or Reject Multiplier
        amt_cb_new_decline_catch = amt_tp_current_tp_new + amt_fn_current_tp_new
        amt_cb_current_decline_catch = amt_tp_current_fn_new + amt_tp_current_tp_new
        amt_cb_reduction_wo_mp = amt_cb_new_decline_catch - amt_cb_current_decline_catch

        tpv_enablement_wo_mp = amt_fp_current_tn_new - amt_tn_current_fp_new
        net_revenue = amt_cb_reduction_wo_mp + tpv_enablement_wo_mp

        # ML system metric split
        count_new_cb = count_tp_current_fn_new + count_fn_current_fn_new
        count_new_reject = count_tp_current_tp_new + count_fn_current_tp_new + count_fp_current_fp_new + count_tn_current_fp_new
        count_new_accept_noncb = count_fp_current_tn_new + count_tn_current_tn_new
        amt_new_cb = amt_tp_current_fn_new + amt_fn_current_fn_new
        amt_new_reject = amt_tp_current_tp_new + amt_fn_current_tp_new + amt_fp_current_fp_new + amt_tn_current_fp_new
        amt_new_accept_noncb = amt_fp_current_tn_new + amt_tn_current_tn_new

        # Client System Split
        count_current_cb = count_fn_current_fn_new + count_fn_current_tp_new
        count_current_reject = count_fp_current_tn_new + count_fp_current_fp_new + count_tp_current_fn_new + count_tp_current_tp_new
        count_current_accept_noncb = count_tn_current_tn_new + count_tn_current_fp_new
        amt_current_cb = amt_fn_current_fn_new + amt_fn_current_tp_new
        amt_current_reject = amt_fp_current_tn_new + amt_fp_current_fp_new + amt_tp_current_fn_new + amt_tp_current_tp_new
        amt_current_accept_noncb = amt_tn_current_tn_new + amt_tn_current_fp_new

        # Calculate the Revenue Metrics: With Fraud or Reject Multiplier
        tpv_current_w_mp = total_amt - reject_multiplier * amt_current_reject - chargeback_multiplier * amt_current_cb
        tpv_new_w_mp = total_amt - reject_multiplier * amt_new_reject - chargeback_multiplier * amt_new_cb
        tpv_enablement_w_mp = tpv_new_w_mp - tpv_current_w_mp

        # cb_caught_efficiency
        cb_caught_efficiency = amt_new_tp / amt_new_fp

        # assign the column one by one
        scores.loc[n_samples, 'current_decline_precision'] = current_decline_precision
        scores.loc[n_samples, 'total_count'] = total_count
        scores.loc[n_samples, 'total_amt'] = total_amt
        scores.loc[n_samples, 'total_cb_count'] = total_cb_count
        scores.loc[n_samples, 'total_cb_amt'] = total_cb_amt
        scores.loc[n_samples, 'min_score_decline'] = min_score_decline
        scores.loc[n_samples, 'count_decline'] = count_decline
        scores.loc[n_samples, 'amt_decline'] = amt_decline
        scores.loc[n_samples, 'prop_data_decline'] = prop_data_decline
        scores.loc[n_samples, 'prop_amt_decline'] = prop_amt_decline
        scores.loc[n_samples, 'count_precision_current'] = count_precision_current
        scores.loc[n_samples, 'count_recall_current'] = count_recall_current
        scores.loc[n_samples, 'count_precision_new'] = count_precision_new
        scores.loc[n_samples, 'count_recall_new'] = count_recall_new
        scores.loc[n_samples, 'amt_precision_current'] = amt_precision_current
        scores.loc[n_samples, 'amt_recall_current'] = amt_recall_current
        scores.loc[n_samples, 'amt_precision_new'] = amt_precision_new
        scores.loc[n_samples, 'amt_recall_new'] = amt_recall_new
        scores.loc[n_samples, 'amt_cb_new_decline_catch'] = amt_cb_new_decline_catch
        scores.loc[n_samples, 'amt_cb_current_decline_catch'] = amt_cb_current_decline_catch
        scores.loc[n_samples, 'count_new_cb'] = count_new_cb
        scores.loc[n_samples, 'count_new_reject'] = count_new_reject
        scores.loc[n_samples, 'count_new_accept_noncb'] = count_new_accept_noncb
        scores.loc[n_samples, 'amt_new_cb'] = amt_new_cb
        scores.loc[n_samples, 'amt_new_reject'] = amt_new_reject
        scores.loc[n_samples, 'amt_new_accept_noncb'] = amt_new_accept_noncb
        scores.loc[n_samples, 'count_current_cb'] = count_current_cb
        scores.loc[n_samples, 'count_current_reject'] = count_current_reject
        scores.loc[n_samples, 'count_current_accept_noncb'] = count_current_accept_noncb
        scores.loc[n_samples, 'amt_current_cb'] = amt_current_cb
        scores.loc[n_samples, 'amt_current_reject'] = amt_current_reject
        scores.loc[n_samples, 'amt_current_accept_noncb'] = amt_current_accept_noncb
        scores.loc[n_samples, 'amt_cb_reduction_wo_mp'] = amt_cb_reduction_wo_mp
        scores.loc[n_samples, 'tpv_enablement_wo_mp'] = tpv_enablement_wo_mp
        scores.loc[n_samples, 'net_revenue'] = net_revenue
        scores.loc[n_samples, 'tpv_enablement_w_mp'] = tpv_enablement_w_mp
        scores.loc[n_samples, 'cb_caught_efficiency'] = cb_caught_efficiency
        scores.loc[n_samples, 'amt_fp_current_tn_new'] = amt_fp_current_tn_new
        scores.loc[n_samples, 'amt_fp_current_fp_new'] = amt_fp_current_fp_new
        scores.loc[n_samples, 'amt_tp_current_fn_new'] = amt_tp_current_fn_new
        scores.loc[n_samples, 'amt_tp_current_tp_new'] = amt_tp_current_tp_new
        scores.loc[n_samples, 'amt_tn_current_tn_new'] = amt_tn_current_tn_new
        scores.loc[n_samples, 'amt_tn_current_fp_new'] = amt_tn_current_fp_new
        scores.loc[n_samples, 'amt_fn_current_fn_new'] = amt_fn_current_fn_new
        scores.loc[n_samples, 'amt_fn_current_tp_new'] = amt_fn_current_tp_new
        scores.loc[n_samples, 'count_fp_current_tn_new'] = count_fp_current_tn_new
        scores.loc[n_samples, 'count_fp_current_fp_new'] = count_fp_current_fp_new
        scores.loc[n_samples, 'count_tp_current_fn_new'] = count_tp_current_fn_new
        scores.loc[n_samples, 'count_tp_current_tp_new'] = count_tp_current_tp_new
        scores.loc[n_samples, 'count_tn_current_tn_new'] = count_tn_current_tn_new
        scores.loc[n_samples, 'count_tn_current_fp_new'] = count_tn_current_fp_new
        scores.loc[n_samples, 'count_fn_current_fn_new'] = count_fn_current_fn_new
        scores.loc[n_samples, 'count_fn_current_tp_new'] = count_fn_current_tp_new
        #print(scores.loc[n_samples, 'count_fn_current_tp_new'])

        #scores.dropna(axis=1, inplace=True)

    return scores


# In[2]:


data = pd.read_csv("Model_performance_toy_data.csv")


# In[8]:

decline_data = data
length = decline_data.shape[0]
# define the number of cutoff point you want to have in the variable num_threshold
num_threshold = 100
n_samples_vec = np.linspace(start=1, stop=10000, num=num_threshold) * length // 10000

# In[13]:


"""Test the two model compare case:"""
two_model_performance_data = model_performance(df=decline_data.copy(), amt='order_total', pred_score='predict_proba', chargeback='cb', current_reject='reject',
                                               n_samples_vec=n_samples_vec, reject_multiplier=0.5, chargeback_multiplier=2, two_models=True, current_decline_precision=0.5)


# In[14]:


print(two_model_performance_data)


# In[16]:


non_decline_data = decline_data.loc[decline_data.reject == 0]
length = non_decline_data.shape[0]
# define the number of cutoff point you want to have in the variable num_threshold
num_threshold = 100
n_samples_vec = np.linspace(start=1, stop=10000, num=num_threshold) * length // 10000

# In[25]:


"""Test the one model compare case:"""
one_model_performance_data = model_performance(df=non_decline_data.copy(), amt='order_total', pred_score='predict_proba', chargeback='cb', current_reject='reject',
                                               n_samples_vec=n_samples_vec, reject_multiplier=0.5, chargeback_multiplier=2, two_models=False, current_decline_precision=0.5)

# In[26]:


print(one_model_performance_data)