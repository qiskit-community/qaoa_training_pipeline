#
#
# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module collects all the methods to train QAOA parameters based on models.

A model is typically based on a collection of pre-trained instances. Then we exploit
parameter transfer properties so that unknown graphs can `inherite` parameters from
the trained or fitted model.
"""
