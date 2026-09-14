# Domain glossary

## Role facts

Observed or extracted statements about what the job primarily delivers and how
the work is performed. Role facts may overlap and must not contain a personal
fit decision.

## Primary deliverable

The main artifact or outcome produced by the role: an AI application, model
engineering, research, AI infrastructure, a non-AI platform, general software,
or unknown.

## Delivery facts

Independent facts about customer contact, product collaboration, embedded
customer implementation, and internal users. These are not mutually exclusive.

## Derived role type

A display label calculated by application rules from role facts. It is not a
fact extracted directly from the job description.

V3 retains this label only as an experiment and compatibility view. V4 removes
it from the decision path because a real job can contain several kinds of work.

## Work-content profile

V4 represents customer implementation, customer advisory, product development,
internal tools, model engineering, research, and AI infrastructure independently.
Each dimension is core, present, or not evidenced. The
application may show all applicable tags; it does not force them into one role
category.

## Candidate policy

The user's personal acceptance and review policy. It converts validated facts
and a derived role type into apply, skip, or human review.

## Location facts versus eligibility

`work_arrangement` records what the JD states: onsite, hybrid, remote, mixed,
or unknown. `candidate_location_eligible` is a separate policy-relative
judgment: whether the stated geography permits this candidate to work from the
San Francisco Bay Area or through compatible US remote work. Keeping these
fields separate prevents the word "remote" from automatically meaning eligible.

## Ground truth

Human-reviewed source facts with supporting quotations. Expected role type and
recommendation are derived from these facts and the candidate policy; they are
not separately hand-entered labels.

## Contract error

Data that violates the declared shape, types, allowed values, or evidence
contract. A plausible combination of role facts is not a contract error merely
because it is unusual.
