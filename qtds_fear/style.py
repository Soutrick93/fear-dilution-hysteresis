# Library module of the qtds_fear package: imported by the scripts in analysis/ and figures/; not run directly.
"""Shared figure style: Springer two-column widths, editable text in PDF (Type 42), colour-blind-safe palette.

Colours carry ecological meaning across every figure:
  crabs (prey) = AQUA, raccoons (mesopredator) = ORANGE, fear / carnivores = VIOLET,
  stable = solid line, unstable = dashed line, bistable region = light blue tint.
"""
import os, pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 84 * MM          # Springer single column
DOUBLE = 174 * MM         # Springer full width

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
MAGENTA, GREEN, VIOLET, RED = '#e87ba4', '#008300', '#6250d6', '#e34948'
INK, INK2, GRID, PAPER = '#1a1a1a', '#52514e', '#e4e3df', '#ffffff'
AXIS = '#9a9993'          # grey axis lines and ticks
TICKTXT = '#4a4945'       # tick labels: dark grey, readable
TINT_BISTABLE = '#dbe8f8'
TINT_COEX = '#e3f4ec'
TINT_FREE = '#fbe5dc'
TINT_CYCLE = '#fbefd0'

PREY, MESO, FEAR = AQUA, ORANGE, VIOLET

# text-light figures by default (labels are added in Illustrator); QTDS_ANNOTATE=1 adds explanatory notes
ANNOTATE = os.environ.get('QTDS_ANNOTATE', '0') == '1'

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_PDF = ROOT / 'figures' / 'pdf'
FIG_PNG = ROOT / 'figures' / 'png'


def apply():
    mpl.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 9.5, 'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5, 'legend.fontsize': 8, 'axes.linewidth': 0.9, 'xtick.major.width': 0.9,
        'ytick.major.width': 0.9, 'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
        'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK,
        'axes.labelpad': 3.0, 'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': TICKTXT,
        'ytick.labelcolor': TICKTXT, 'text.color': INK, 'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round',
        'lines.dash_capstyle': 'round', 'legend.frameon': False, 'figure.dpi': 150, 'savefig.dpi': 450,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
        'image.interpolation': 'antialiased', 'axes.axisbelow': True,
    })


def tidy(ax):
    """Grey, slightly offset axes (2-D axes only)."""
    if getattr(ax, 'name', '') == '3d':
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.line.set_color(AXIS); a.set_tick_params(colors=AXIS, labelcolor=TICKTXT)
        return
    for side, sp in ax.spines.items():
        if sp.get_visible():
            sp.set_color(AXIS); sp.set_linewidth(0.9); sp.set_position(('outward', 3))
    ax.tick_params(colors=AXIS, labelcolor=TICKTXT, which='both')


def panel_label(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left', color=INK)


def save(fig, name):
    for ax in fig.axes:
        if ax.get_label() != '<colorbar>':
            tidy(ax)
        else:
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(colors=AXIS, labelcolor=TICKTXT)
    FIG_PDF.mkdir(parents=True, exist_ok=True); FIG_PNG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF / f'{name}.pdf', bbox_inches='tight', pad_inches=0.08)  # margin: labels ending in subscripts are not clipped
    fig.savefig(FIG_PNG / f'{name}.png', bbox_inches='tight', pad_inches=0.08, dpi=300)
    plt.close(fig)
    return FIG_PDF / f'{name}.pdf'


# ---- text that is only drawn in annotated mode (QTDS_ANNOTATE=1); arrows without text are always drawn
def note(ax, *args, **kw):
    if ANNOTATE:
        return ax.text(*args, **kw)


def anno(ax, text, *args, **kw):
    if ANNOTATE or text == '':
        return ax.annotate(text, *args, **kw)


def title(ax, text, **kw):
    if ANNOTATE:
        return ax.set_title(text, **kw)
