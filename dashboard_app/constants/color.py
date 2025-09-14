from enum import Enum


# TODO: use a theme instead
class Color(str, Enum):
    PRIMARY = "rgb(74, 117, 240)"  # Blue
    SECONDARY = "rgb(224, 105, 99)"  # Red
    BACKGROUND = "black"
    TEXT = "white"
    SUCCESS = "rgb(84, 199, 191)"  # Teal-Green
    WARNING = "rgb(224, 167, 99)"  # Orange
    DANGER = "rgb(224, 105, 99)"  # Red
    HIGHLIGHT = "rgb(196, 199, 28)"  # Yellow
    MUTED = "rgb(186, 184, 184)"  # Grey
    BORDER = "black"
    ANNOTATION_BG = "white"
    ANNOTATION_TEXT = "black"
