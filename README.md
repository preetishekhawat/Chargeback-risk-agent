# FEMA — Chargeback / Dispute Risk Scorer
### Razorpay Buildathon — Track 2: AI Risk Manager

## What this is

FEMA looks at a transaction *before* any dispute has happened, and
estimates how likely it is to turn into a chargeback later — based on
payment details, order status, and delivery timing. Instead of trying
to catch every risky transaction (which the data honestly doesn't
support well — more on that below), it prioritizes the riskiest 10%
for human review and auto-clears the rest, logging every decision
along the way.

This isn't a "we trained a model and got a great score" project. The
model's predictive power is genuinely modest, and rather than hide
that or chase a prettier number, this repo documents an actual
investigation into *why* — five separate hypotheses tested, one false
lead caught and corrected, and an honest final answer.

## Architecture

```
Transaction arrives
        |
        v
Feature Engineering
        |
        v
Risk Engine (Logistic Regression) → risk probability
        |
        v
Decision Engine (top 10% by risk → escalate, rest → resolve)
        |
        v
Audit Trail (every decision logged and traceable)
```

## Setup

```
pip install -r requirements.txt
```

## The data

This uses the **v1** version of the "Razorpay Chargeback AI Synthetic
Dataset" from Kaggle — specifically the `razorpay_dispute_ai_dataset_v1`
folder, not the other bundled export (its `transactions.csv` is
missing the key needed to join to orders — cost some hours figuring
that out, documented below).

Download `orders.csv`, `transactions.csv`, and `disputes.csv` from
that folder and place them directly in `data/raw/` (flat, no
subfolders). You don't need `customers.csv`, `merchants.csv`, or
`deliveries.csv` — they were checked and don't add anything the
pipeline uses.

## Running it

**Core pipeline** — this is what actually produces the model and demo:

```
python src/pipeline/load_data.py       # joins transactions + orders + disputes, builds the label
python src/pipeline/build_features.py  # encodes categoricals, adds derived features
python src/pipeline/train_model.py     # trains the model, runs the decision layer, saves the audit trail
streamlit run src/pipeline/app.py      # interactive demo — pick a transaction, see its score and decision
```

**Investigation** — this is the part I'd actually point you to first if
you want to understand the project, not just run it:

```
python src/investigation/diagnose_timing.py         # checks disputes aren't leaking post-event info
python src/investigation/diagnose_target.py         # class balance, duplicates, subgroup dispute rates
python src/investigation/build_customer_history.py  # prior-only customer behavior features
python src/investigation/build_merchant_history.py  # prior-only merchant behavior features
python src/investigation/compare_models.py          # tests all of the above, plus a nonlinear model
python src/investigation/business_metrics.py         # precision/recall/lift at a 10% review budget
```

## What I found

The first model — transaction and order features feeding a Logistic
Regression — came back with a ROC-AUC of about 0.546. Barely above a
coin flip. I could've reported that and stopped, but I wanted to know
whether it was a fixable problem or a real limit of the data.

**Was it leaking future information?** No — I checked when disputes
actually get filed relative to delivery, and it turns out 100% of them
happen *after* delivery, about two weeks later on average. So the
delivery-related features were legitimately known beforehand.

**Was customer history the missing piece?** I built features like
"how many orders has this customer placed before, what's their average
spend, how often have they cancelled" — all computed using only orders
that happened *before* the current one, so there's no leakage. Result:
essentially no improvement (AUC 0.506, basically random on its own).

**What about merchant history?** This one had a twist. Looking at raw
merchant-level dispute rates, some merchants sat at 25–33% versus a
9.6% baseline — that looked like a real, strong pattern. But those
merchants only had 20–40 transactions each, and with that little data,
extreme-looking rates happen by chance alone. When I built the same
kind of prior-only merchant feature and tested it properly, it came
back at 0.496 — no better than random. The apparent pattern was
statistical noise, not signal, and I'm glad I tested it instead of
building the whole story around it.

**Was the model just too simple?** I ran a Random Forest — a
nonlinear model that can pick up interactions a linear model can't —
on the same combined features. It landed at 0.549. Statistically the
same as the linear model. So it wasn't a model-capacity problem
either.

**Is 0.546 even a real number, or just noise?** I bootstrapped a 95%
confidence interval: [0.536, 0.561]. That excludes 0.50, so there is a
real, if small, signal here — it's just genuinely weak, not hidden
somewhere I hadn't looked.

## What this means in practice

Rather than pretend the model is stronger than it is, FEMA is built
around using that weak-but-real signal as efficiently as possible: if
a merchant can only manually review 10% of transactions, prioritize
the ones the model flags as riskiest. Reviewing at random would catch
about 10% of disputes; FEMA's prioritized review catches about 11.7%
— a modest 1.17x lift, and an honest one.

## What I'd do with more time

The dataset includes a text field describing each dispute's claim,
which is probably where the real predictive signal lives — but it
only exists *after* a dispute is filed, so it can't be used for a
before-the-fact prediction task without reframing the whole problem
(e.g., as a post-dispute evidence classifier instead). That's the
clearest next step, not another round of feature engineering on the
same structured columns.

## Known limitations

- Transactions from the same order can end up split across train and
  test — a mild potential leak via shared order-level context, not
  fully addressed given time constraints.
- The 10% review threshold is framed as a review-capacity constraint,
  not a cost-based one — I didn't want to invent dollar figures for
  false-positive/false-negative cost without real cost data to back
  them up.