"""
report.py
---------
Combined actuarial summary report.

Ties together:
    - Reserving: IBNR by accident year (chain ladder + BF)
    - Pricing: pure premium by risk segment (freq × severity GLM)
    - Summary: indicated rate change, reserve adequacy, combined ratio

TODO: implement ActuarialReport class
TODO: implement combined_analysis() end-to-end
"""
