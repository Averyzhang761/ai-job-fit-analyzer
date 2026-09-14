# Domain glossary

## Responsibility

An explicit statement about work the role performs. The current pipeline stores the exact source
quotation and one or more activity tags for each responsibility. A real role can
contain several activities, so responsibilities are not collapsed into one
exclusive role category.

## Work activity

A reusable tag attached to a source-backed responsibility, such as customer
implementation, customer advisory, product development, internal tools, model
engineering, research, AI infrastructure, or general software. Tags describe
work present in the JD; they are not importance scores or personal-fit decisions.

## Candidate policy

The user's personal acceptance and review policy. It combines validated work
activities with candidate-specific location and work-authorization constraints
to produce `apply`, `skip`, or `human_review`.

## Location facts versus eligibility

`work_arrangement`, `stated_locations`, and `remote_scope` record source facts.
Candidate location eligibility is derived directly for unambiguous US/global
remote cases and uses a focused reviewer only when those facts are insufficient.

## Work authorization

The source contract records explicit sponsorship and citizenship/green-card
statements. Python derives candidate eligibility. Sponsorship is not treated as
proof of H-1B transfer support unless the JD states that narrower fact.

## Evidence boundary

Pydantic verifies that every returned responsibility quotation exists verbatim
in the original JD. This establishes evidence existence, not extraction
completeness or activity-tag correctness.

## Ground truth

Human-reviewed source facts with supporting quotations. Provisional labels and
model-generated drafts are diagnostic material, not accepted ground truth.

## Contract error

Data that violates the declared shape, types, allowed values, or evidence
contract. Uncertainty and overlapping responsibilities are valid domain states.
