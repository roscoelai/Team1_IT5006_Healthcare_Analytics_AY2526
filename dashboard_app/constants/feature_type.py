from enum import Enum


class FeatureType(str, Enum):
    IDENTIFIER = "Identifier"
    TEXT = "Text"
    CONTINUOUS = "Continuous"
    DISCRETE = "Discrete"
    NOMINAL = "Nominal"
    ORDINAL = "Ordinal"
    DATETIME = "Datetime"
    BOOLEAN = "Boolean"
