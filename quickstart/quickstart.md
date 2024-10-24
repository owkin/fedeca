## Quickstart
This quickstart assumes users have already installed fedeca in a conda environment.
FedECA tries to mimic scikit-learn API as much as possible with the constraints
of distributed learning.
The first step in data science is always the data.
We need to first use or generate some survival data in pandas.dataframe format.
Note that fedeca should work on any data format, provided that the
return type of the substra opener is indeed a pandas.dataframe but let's keep
it simple in this quickstart.

We recommend users to first install ipython (`pip install ipython`) or jupyter,
and to copy-paste and run the content of the blocks sequentially either in the
ipython shell or in a jupyter notebook.

(Don't forget to make sure the `ipython` being called is the one from the fedeca
conda environment by calling `which ipython`. In case it is not the correct one
running `hash -r` usually does the trick.)


Here we will use fedeca utils which will generate some synthetic survival data
following CoxPH assumptions:

```python
import pandas as pd
from fedeca.utils.survival_utils import CoxData
# Let's generate 1000 data samples with 10 covariates
data = CoxData(seed=42, n_samples=1000, ndim=10)
df = data.generate_dataframe()

# We remove the true propensity score
df = df.drop(columns=["propensity_scores"], axis=1)
```
Let's inspect the data that we have here.
```python
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 13 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   X_0        1000 non-null   float64
#  1   X_1        1000 non-null   float64
#  2   X_2        1000 non-null   float64
#  3   X_3        1000 non-null   float64
#  4   X_4        1000 non-null   float64
#  5   X_5        1000 non-null   float64
#  6   X_6        1000 non-null   float64
#  7   X_7        1000 non-null   float64
#  8   X_8        1000 non-null   float64
#  9   X_9        1000 non-null   float64
#  10  time       1000 non-null   float64
#  11  event      1000 non-null   uint8
#  12  treatment  1000 non-null   uint8
# dtypes: float64(11), uint8(2)
# memory usage: 88.0 KB
print(df.head())
#         X_0       X_1       X_2       X_3       X_4       X_5       X_6       X_7       X_8       X_9      time  event  treatment
# 0 -0.918373 -0.814340 -0.148994  0.482720 -1.130384 -1.254769 -0.462002  1.451622  1.199705  0.133197  2.573516      1          1
# 1  0.360051 -0.863619  0.198673  0.330630 -0.189184 -0.802424 -1.694990 -0.989009 -0.421245 -0.112665  0.519108      1          1
# 2  0.442502  0.024682  0.069500 -0.398015 -0.521236 -0.824907  0.373018  1.016843  0.765661  0.858817  0.652803      1          1
# 3 -0.783965 -1.116391 -1.482413 -2.039827 -1.639304 -0.500380 -0.298467 -1.801688 -0.743004 -0.724039  0.074925      1          1
# 4 -0.199620 -0.652347 -0.018776  0.004630 -0.122242 -0.413490 -0.450718 -0.761894 -1.323135 -0.234899  0.006951      1          1
print(df["treatment"].unique())
# array([1, 0], dtype=uint8)
df["treatment"].sum()
# 500
```
So we have survival data with covariates and a binary treatment variable.
Let's inspect it using proper survival plots using the great survival analysis
package [lifelines](https://github.com/CamDavidsonPilon/lifelines) that was a
source of inspiration for fedeca:
```python
from lifelines import KaplanMeierFitter as KMF
import matplotlib.pyplot as plt
treatments = [0, 1]
kms = [KMF().fit(durations=df.loc[df["treatment"] == t]["time"], event_observed=df.loc[df["treatment"] == t]["event"]) for t in treatments]

axs = [km.plot(label="treated" if t == 1 else "untreated") for km, t in zip(kms, treatments)]
axs[-1].set_ylabel("Survival Probability")
plt.xlim(0, 1500)
plt.savefig("treated_vs_untreated.pdf", bbox_inches="tight")
```
Open `treated_vs_untreated.pdf` in your favorite pdf viewer and see for yourself.

## Pooled IPTW analysis
The treatment seems to improve survival but it's hard to say for sure as it might
simply be due to chance or sampling bias.
Let's perform an IPTW analysis to be sure:

```python
from fedeca.competitors import PooledIPTW
pooled_iptw = PooledIPTW(treated_col="treatment", event_col="event", duration_col="time")
# Targets is the propensity weights
pooled_iptw.fit(data=df, targets=None)
print(pooled_iptw.results_)
#                coef  exp(coef)  se(coef)  coef lower 95%  coef upper 95%  exp(coef) lower 95%  exp(coef) upper 95%  cmp to         z         p  -log2(p)
# covariate
# treatment  0.041727    1.04261  0.070581       -0.096609        0.180064             0.907911             1.197294     0.0  0.591196  0.554389   0.85103
```
When looking at the `p-value=0.554389 > 0.05`, thus judging by what we observe we
cannot say for sure that there is a treatment effect. We say the ATE is non significant.

## Distributed Analysis

However in practice data is private and held by different institutions. Therefore
in practice each client holds a subset of the rows of our dataframe.
We will simulate this using a realistic scenario where a "pharma" node is developing
a new drug and thus holds all treated and the rest of the data is split across
3 other institutions where patients were treated with the old drug.
We will use the split utils of FedECA.
```python
from fedeca.utils.data_utils import split_dataframe_across_clients

clients, train_data_nodes, _, _, _ = split_dataframe_across_clients(
    df,
    n_clients=4,
    split_method= "split_control_over_centers",
    split_method_kwargs={"treatment_info": "treatment"},
    data_path="./data",
    backend_type="simu",
)
```
Note that you can replace split_method by any callable with the signature
`pd.DataFrame -> list[list[int]]` where the list of list of ints is the split of the indices
of the df across the different institutions.
To convince you that the split was effective you can inspect the folder "./data".
You will find different subfolders `center0` to `center3` each with different
parts of the data.
To unpack a bit what is going on in more depth, we have created a dict of client
'clients',
which is a dict with 4 keys containing substra API handles towards the different
institutions and their data.
`train_data_nodes` is a list of handles towards the datasets of the different institutions
that were registered through the substra interface using the data in the different
folders.
You might have noticed that we did not talk about the `backend_type` argument. 
This argument is used to choose on which network will experiments be run.
"simu" means in-RAM. If you finish this tutorial do try other values such as:
"docker" or "subprocess" but expect a significant slow-down as experiments
get closer and closer to a real distributed system.

Now let's try to see if we can reproduce the pooled anaysis in this much more
complicated distributed setting:
```python
from fedeca import FedECA
# We use the first client as the node, which launches order
ds_client = clients[list(clients.keys())[0]]
fed_iptw = FedECA(ndim=10, ds_client=ds_client, train_data_nodes=train_data_nodes, treated_col="treatment", duration_col="time", event_col="event", variance_method="robust")
fed_iptw.run()
# Final partial log-likelihood:
# [-11499.19619422]
#        coef  se(coef)  coef lower 95%  coef upper 95%         z         p  exp(coef)  exp(coef) lower 95%  exp(coef) upper 95%
# 0  0.041718  0.070581       -0.096618        0.180054  0.591062  0.554479     1.0426             0.907902             1.197282
```
In fact what we did above is both quite verbose. For simulation purposes we
advise to use directly the scikit-learn inspired syntax:
```python
fed_iptw = FedECA(ndim=10, treated_col="treatment", event_col="event", duration_col="time")
fed_iptw.fit(df, n_clients=4, split_method="split_control_over_centers", split_method_kwargs={"treatment_info": "treatment"}, data_path="./data", variance_method="robust", backend_type="simu")
#        coef  se(coef)  coef lower 95%  coef upper 95%         z         p  exp(coef)  exp(coef) lower 95%  exp(coef) upper 95%
# 0  0.041718  0.070581       -0.096618        0.180054  0.591062  0.554479     1.0426             0.907902             1.197282
```
We find a similar p-value ! The distributed analysis is working as expected.
We recommend to users that made it to here as a next step to use their own data
and write custom split functions and to test this pipeline under various
heterogeneity settings.
Another interesting avenue is to try adding differential privacy to the training
of the propensity model but that is outside the scope of this quickstart. 



