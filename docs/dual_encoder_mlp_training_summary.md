# Dual Encoder MLP Training Summary

## Scope
This note summarizes the contrastive MLP dual-encoder training and tuning effort for the CITE-seq project. It covers the observed failure modes, the fixes that were applied, what those fixes changed, how to interpret the validation curves, and why the model remains useful even when Stage A validation looks imperfect.

## Short Version
The MLP dual encoder is working. The main issues were not a broken model or obvious test leakage, but a combination of:

- validation split quality and class coverage,
- contrastive-loss dynamics that are not expected to mirror training loss tightly,
- and an AUROC edge case when some classes were missing from a split.

After fixing the split policy and the AUROC computation path, the model produced strong final test results and stable seed behavior.

## What The Model Does
The MLP dual encoder has two stages:

1. Stage A: contrastive pretraining.
   - RNA and protein go through separate MLP encoders.
   - CLIP-style contrastive loss aligns the two modalities.
   - Validation is used for checkpoint selection and early stopping.

2. Stage B: classifier fine-tuning.
   - The encoders are frozen.
   - A classifier is trained on concatenated embeddings.
   - Validation is used to select the best classifier checkpoint.
   - Final test is only evaluated after model selection is complete.

## What Went Wrong Initially
### 1. Stage A validation looked flat or ugly
The Stage A validation loss often stayed high and moved only slightly while the training loss fell much faster.

This looked suspicious at first, but it was not evidence of test leakage by itself.

### 2. Final macro AUROC sometimes appeared as NaN
The training code was computing AUROC on the final test split, but the logic could fail when the held-out split was missing one or more classes. In that case the metric path could fall back to NaN/null even though the model predictions were fine.

### 3. The validation split was not always representative enough
Even when validation contained most classes, the split could still be noisy or class-imbalanced enough that Stage A validation loss did not follow training loss closely.

## What Was Changed
### Split handling
The validation split was made train-only by default, rather than relying on an implied predefined validation cohort.

That change helped in two ways:

- it kept the final test split isolated,
- and it made validation selection less dependent on a fragile predefined split.

### Class-aware split construction
The train-derived validation split was improved so that it tries to keep class coverage more balanced.

This matters because contrastive and classifier metrics become unstable when rare classes are underrepresented.

### AUROC robustness
The final AUROC computation was hardened so it handles missing classes more safely.

The key idea was:
- restrict AUROC to the classes present in the split,
- renormalize probabilities after subsetting,
- and avoid treating a missing-class split as a normal multiclass case.

### Debug workflow
A dedicated debug Slurm script was added to print:

- total class counts,
- train class counts,
- validation class counts,
- test class counts,
- Stage A train/val loss behavior,
- and final metrics.

That script made it much easier to see whether the issue was split quality or training dynamics.

## What Led To What
### Better split coverage led to more meaningful validation
Once the validation split was large enough and class-covered, Stage A validation became interpretable even if it was still not smooth.

### AUROC fix led to numeric final metrics
Once the final test metric path handled missing classes correctly, final macro AUROC stopped disappearing as NaN.

### Stage B remained strong even when Stage A looked imperfect
The classifier stage benefited from the learned representation even though Stage A validation loss was not a perfect mirror of Stage A training loss.

That is normal here. The contrastive objective is not identical to downstream classification, so the curves do not need to move in lockstep.

## Why Stage A Validation Looks Like That
Stage A validation loss being flatter than training loss is plausible for several reasons.

### 1. Contrastive loss is batch-sensitive
Contrastive learning depends on in-batch negatives. Small differences in batch composition can affect validation loss a lot.

### 2. Validation is noisier than training
Training had input dropout and Gaussian noise, while validation did not. That creates a mismatch in scale and behavior.

### 3. The objective is indirect
Stage A is optimizing modality alignment, not final class accuracy. Good representation learning does not have to produce a perfectly decreasing validation curve.

### 4. Long-tail classes make the split harder
Some classes are rare. Even if they appear in validation, they can still make the curve look noisy or less smooth.

### 5. Stage A and Stage B measure different things
Stage A validation loss is a proxy for representation quality. Stage B validation accuracy/AUROC is closer to the actual downstream goal.

So a Stage A curve that looks awkward is not, by itself, a sign that the model is useless.

## Possible Causes Of The Stage A Shape
The observed Stage A curve shape can come from one or more of the following:

- validation split not perfectly representative,
- class imbalance in the held-out split,
- contrastive batch effects,
- stronger train-time regularization than val-time behavior,
- learning-rate schedule effects,
- and the gap between representation learning and downstream classification.

In this repository, the evidence pointed more toward optimization dynamics and split quality than toward leakage.

## Why The Model Is Still Useful
The final results show that the model learned a strong representation.

What matters most:

- final test accuracy was high,
- final macro AUROC was high once the metric path was fixed,
- Stage B validation was stable,
- and seed behavior looked consistent.

That means the representation learned by Stage A is useful even if Stage A validation loss is not perfectly smooth.

A model can be useful when:

- the contrastive objective improves representation structure,
- the downstream classifier performs well,
- and the final metrics remain stable across runs.

That is what happened here.

## Interpretation Of The Final Runs
The final debug runs showed:

- strong final test accuracy,
- strong final macro AUROC,
- Stage A training loss falling steadily,
- Stage A validation loss staying higher but not collapsing,
- and Stage B curves that looked substantially healthier.

That combination is consistent with a usable contrastive dual encoder, not a broken one.

## Practical Lessons
1. Do not judge Stage A only by the shape of validation loss.
2. Always inspect class counts for train, validation, and test.
3. Make final metric code robust to missing classes.
4. Keep the test split isolated until the very end.
5. Use a debug run that prints split coverage and summary metrics before tuning too aggressively.

## Recommended Debug Checklist
When the curves look strange, check these first:

- Does validation contain enough examples per class?
- Is the final test missing any classes?
- Are Stage A train and val behaving differently because of regularization or batch effects?
- Is AUROC being computed on a split that cannot support multiclass AUROC cleanly?
- Are you comparing Stage A validation loss to the wrong success metric?

## Bottom Line
The MLP dual encoder is not fundamentally broken.

The main issues were:

- validation split quality,
- AUROC handling for missing-class splits,
- and the fact that Stage A validation loss is a noisy proxy for downstream utility.

Once those were addressed, the model produced strong downstream performance and became a reasonable, useful training setup.
