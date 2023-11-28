"""Utilities for visualisation."""
import matplotlib.pyplot as plt
from matplotlib.offsetbox import VPacker


def adjust_legend_subtitles(legend):
    """Make invisible-handle "subtitles" entries look more like titles.

    Adapted from:
        https://github.com/mwaskom/seaborn/blob/63d91bf0298009effc9de889f7f419f59deb3960/seaborn/utils.py#L830
    Reference for `offsite_points`:
        https://stackoverflow.com/questions/24787041/multiple-titles-in-legend-in-matplotlib
        https://github.com/matplotlib/matplotlib/blob/3180c94d84f4aeb8494e3fde9e39f5f7e4e244b6/lib/matplotlib/legend.py#L926
    """
    # Legend title not in rcParams until 3.0
    title_font_size = plt.rcParams.get("legend.title_fontsize", None)
    offset_points = legend._fontsize * legend.handletextpad
    hpackers = legend.findobj(VPacker)[0].get_children()
    for hpack in hpackers:
        draw_area, text_area = hpack.get_children()
        handles = draw_area.get_children()
        if not all(artist.get_visible() for artist in handles):
            draw_area.set_width(-offset_points)
            for text in text_area.get_children():
                if title_font_size is not None:
                    text.set_size(title_font_size)
