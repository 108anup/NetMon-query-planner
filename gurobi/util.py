# https://jwalton.info/Embed-Publication-Matplotlib-Latex/

def get_fig_size(col=1, height_frac=1):
    width = LATEX_LINE_WIDTH_IN * col
    height = width * GOLDEN_RATIO * height_frac
    return (width, height)


def pt2in(pt):
    return pt/72.72


LATEX_TEXT_WIDTH_PT = 505.89
LATEX_TEXT_WIDTH_IN = pt2in(LATEX_TEXT_WIDTH_PT)
LATEX_LINE_WIDTH_PT = 241.02039
LATEX_LINE_WIDTH_IN = pt2in(LATEX_LINE_WIDTH_PT)
GOLDEN_RATIO = (5**.5 - 1) / 2
