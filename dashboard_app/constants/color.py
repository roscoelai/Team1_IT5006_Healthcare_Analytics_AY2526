from enum import Enum


# TODO: use a theme instead
class Color(str, Enum):
    PRIMARY = "rgb(74, 117, 240)"
    SECONDARY = "indianred"
    BACKGROUND = "black"
    TEXT = "white"
    SUCCESS = "rgb(84, 199, 191)"
    WARNING = "rgb(224, 167, 99)"
    DANGER = "rgb(224, 105, 99)"
    HIGHLIGHT = "rgb(196, 199, 28)"
    MUTED = "rgb(186, 184, 184)"
    BORDER = "black"
    ANNOTATION_BG = "white"
    ANNOTATION_TEXT = "black"
