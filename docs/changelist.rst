Change List
===========

v1.0.2
######

- Added GridBasedWindField and GenerativeWindFieldSampler.
- Deprecated GenerativeWindField in favor of GridBasedWindField and
  GenerativeWindFieldSampler.
- Bug fix: previously the train script would load the latest checkpoint
  when restarting but then resume from the previous iteration. It is now
  fixed, so if reloading checkpoint i, the agent will continue working
  on iteration i + 1.
- Vectorized wind column feature calculations. This gives about a 16% speed
  increase.
- Cleanups in balloon.py, by removing staticfunctions.
- Moved wind field creation to run_helpers, since it is common for
  training and evaluation.
- Added a flag for calculating the flight path to eval_lib, to allow for
  faster evaluation if you don't need flight paths.
- Improvements to AcmeEvalAgent by making it more configurable.
