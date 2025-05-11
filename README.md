# Correlation Filter Tracking

**Author:** Ana Poklukar

**Date:** April 2025

---

This project was developed for the **Advanced Computer Vision Methods** course at the University of Ljubljana. It implements and evaluates the **MOSSE (Minimum Output Sum of Squared Error) correlation filter tracker**, with performance assessed using the [VOT2014](https://www.votchallenge.net/vot2014/) benchmark. The work explores the effects of key parameters on tracking accuracy, robustness, and runtime, and compares initialization and per-frame processing times across different sequences.

### Repository Structure

* `mosse_tracker.py`: Core implementation of the MOSSE correlation filter tracker, compatible with the [**Tracking Toolkit Lite**](https://github.com/alanlukezic/pytracking-toolkit-lite) framework.
* `report.pdf`: Comprehensive report detailing the algorithm, experimental setup, results, and conclusions.
