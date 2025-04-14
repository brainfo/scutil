import scanpy as sc
import seaborn as sns
from matplotlib.pyplot import rc_context
from typing import Dict, Optional
from collections import defaultdict
import matplotlib
import seaborn as sns
from collections import Counter
import matplotlib
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set_theme(style="ticks", rc=custom_params)
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.set_figure_params(dpi=900)
## suppress warnings
import warnings
warnings.filterwarnings('ignore')
## no patch outline for matplotlib
matplotlib.rcParams['figure.edgecolor'] = 'none'
matplotlib.rcParams['lines.linewidth'] = 0