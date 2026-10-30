"""Folder layout of the study, relative to this file: <root>/paper (scripts, results, manuscript) and <root>/_dev
(helper modules; sfbay_fine/ holds the bay inputs). Set CAD_ROOT to use another root."""
import os

ROOT = os.environ.get("CAD_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
PAPER = os.path.join(ROOT, "paper")
DEV = os.path.join(ROOT, "_dev")
SFBAY = os.path.join(DEV, "sfbay_fine")
FIGURES = os.path.join(PAPER, "figures")
