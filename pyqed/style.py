from matplotlib import rc, ticker
import matplotlib.pyplot as plt
# import proplot as plt

import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pickle


def read_result(fname):
    '''
    read result obj saved with pickle
    '''
    with open(fname, 'rb') as f:
        result = pickle.load(f)
    return result


def subplots(nrows=1, ncols=1, figsize = (4, 3), sharex=True, \
             sharey=True, **kwargs):

    if nrows == 1 and ncols == 1:

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, **kwargs)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(direction='in', length=6)

        return fig, ax

    elif (nrows > 1 and ncols==1) or (ncols > 1 and nrows==1):

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, \
                                sharey=sharey, **kwargs)

        for ax in axs:
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
            ax.tick_params(direction='in', length=6, labelsize=20)


        return fig, axs

    else:

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, \
                               sharex=True, sharey=True)

        return fig, ax

def curve(x, y, **kwargs):
    """
    simple 1D curve plot

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = subplots()
    ax.plot(x, y, **kwargs)

    return fig, ax

def set_style(fontsize=12):

    size = fontsize

    fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
    'weight' : 'normal', 'size' : fontsize}

    # fontProperties = {'family':'sans-serif','sans-serif':['Arial'],
    # 'weight' : 'normal', 'size' : fontsize}

    rc('font', **fontProperties)

    rc('text', usetex=False)

    mpl.rcParams['mathtext.rm'] = 'Computer Modern'
    mpl.rcParams['mathtext.it'] = 'Computer Modern:italic'
    mpl.rcParams['mathtext.bf'] = 'Computer Modern:bold'

    plt.rc('xtick', color='k', labelsize='large', direction='in')
    plt.rc('ytick', color='k', labelsize='large', direction='in')
    plt.rc('xtick.major', size=6, pad=6)
    plt.rc('xtick.minor', size=4, pad=6)
    plt.rc('ytick.major', size=6, pad=6)
    plt.rc('ytick.minor', size=4, pad=6)
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


#     plt.rcParams['text.latex.preamble'] = [
# ##    r'\usepackage{time}',
#     r'\usepackage{tgheros}',    # helvetica font
#     r'\usepackage[]{amsmath}',   # math-font matching  helvetica
#     r'\usepackage{bm}',
# #    r'\sansmath'                # actually tell tex to use it!
#     r'\usepackage{siunitx}',    # micro symbols
#     r'\sisetup{detect-all}',    # force siunitx to use the fonts
#     ]

    #rc('text.latex', preamble=r'\usepackage{cmbright}')


    plt.rc('axes', titlesize=size)  # fontsize of the axes title
    plt.rc('axes', labelsize=size)  # fontsize of the x any y labels
    # plt.rc('xtick', labelsize=size)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=size)  # legend fontsize
    plt.rc('figure', titlesize=size)  # # size of the figure title
    plt.rc('axes', linewidth=1)

    #plt.rcParams['axes.labelweight'] = 'normal'

    plt.locator_params(axis='y')

    # the axes attributes need to be set before the call to subplot
    #plt.rc('xtick.major', size=4, pad=4)
    #plt.rc('xtick.minor', size=3, pad=4)
    #plt.rc('ytick.major', size=4, pad=4)
    #plt.rc('ytick.minor', size=3, pad=4)

    plt.rc('savefig',dpi=120)

    plt.legend(frameon=False)

    # matlab rgb line colors
    linecolors = [ (0,    0.4470,    0.7410),
    (0.8500,  0.3250,    0.0980),
    (0.9290,  0.6940,    0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840)]

    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=linecolors)
    #plt.rcParams["xtick.minor.visible"] =  True


    # using aliases for color, linestyle and linewidth; gray, solid, thick
    #plt.rc('grid', c='0.5', ls='-', lw=5)
    plt.rc('lines', lw=2)
    return

def matplot(x, y, f, vmin=None, vmax=None, ticks=None, output='output.pdf', xlabel='X', \
            ylabel='Y', diverge=False, cmap='viridis', **kwargs):
    """

    Parameters
    ----------
    f : 2D array
        array to be plotted.

    extent: list [xmin, xmax, ymin, ymax]

    Returns
    -------
    Save a fig in the current directory.

    To be deprecated. Please use imshow.
    """

    fig, ax = plt.subplots(figsize=(4,3))

    set_style()

    if diverge:
        cmap = "RdBu_r"
    else:
        cmap = 'viridis'

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    extent = [xmin, xmax, ymin, ymax]
    cntr = ax.imshow(f.T, aspect='auto', cmap=cmap, extent=extent, \
                      origin='lower', vmin=vmin, vmax=vmax, **kwargs)

    ax.set_aspect('auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.colorbar(cntr, ticks=ticks)

    ax.xaxis.set_ticks_position('bottom')

#    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.14, left=0.14, top=0.96, right=0.94)
    if output is not None:
        fig.savefig(output, dpi=1200)

    return fig, ax

def imshow(x, y, f, vmin=None, vmax=None, ticks=None, output='output.pdf', xlabel='X', \
            ylabel='Y', diverge=False, cmap='viridis', **kwargs):
    """

    Parameters
    ----------
    f : 2D array
        array to be plotted.

    extent: list [xmin, xmax, ymin, ymax]

    Returns
    -------
    Save a fig in the current directory.

    """

    fig, ax = plt.subplots(figsize=(4,3))

    set_style()

    if diverge:
        cmap = "RdBu_r"
    else:
        cmap = 'viridis'

    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    extent = [xmin, xmax, ymin, ymax]
    cntr = ax.imshow(f.T, aspect='auto', cmap=cmap, extent=extent, \
                      origin='lower', vmin=vmin, vmax=vmax, **kwargs)

    ax.set_aspect('auto')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.colorbar(cntr, ticks=ticks)

    ax.xaxis.set_ticks_position('bottom')

#    fig.subplots_adjust(wspace=0, hspace=0, bottom=0.14, left=0.14, top=0.96, right=0.94)
    if output is not None:
        fig.savefig(output, dpi=1200)

    return fig, ax

def color_code(x, y, z, fig, ax, cbar=False):
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(z)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    # if cbar:
    #     cbar = fig.colorbar(line, orientation='horizontal')
    #     cbar.set_ticks([0., 1.])
    #     cbar.set_ticklabels(['matter', 'photon'])

    # ax.set_xlim(-6,4)
    # ax.set_ylim(3.,8.0)

    return line


def level_scheme(E, ylim=None, fname=None):
    """
    plot the energy levels
    Parameters
    ----------
    E

    Returns
    -------

    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(3,6))
    ax.set_frame_on(False)     # Alternate way to turn frame off

    ax.hlines(y=E, xmin=0, xmax=0.1, lw=2)
    ax.set_ylabel('Energy (eV)')
    ax.set_ylim(ylim)

    ax.axes.get_xaxis().set_visible(False)  # xticks off

    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))

    fig.subplots_adjust(left=0.28, right=0.98)
    if fname is not None:
        fig.savefig(fname)

    plt.show()
    return ax


def two_scales(x, yl, yr, xlabel=None, ylabels=None, xlim=None, yllim=None, yrlim=None,\
               yticks=None, fname='output.pdf'):
    fig, ax = subplots()
    ax.plot(x, yl, '-s')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabels[0])
    # ax.set_ylabel()
    if yllim is not None:
        ax.set_ylim(yllim)


    if xlim is not None:
        ax.set_xlim(xlim)

    ax2 = ax.twinx()
    ax2.plot(x, yr, 'r-o')
    ax2.set_ylabel(ylabels[1], color='red')
    ax2.tick_params(axis='y', labelcolor='r')



    if yrlim is not None:
        ax2.set_ylim(yrlim)

    if yticks is not None:
        ax2.set_yticks(yticks)

    fig.savefig(fname)
    return

def surf(x, y, f, fname='output.png', xlabel='X', \
         ylabel='Y', zlabel='Z', cmap=None, title=None, method='matplotlib'):

    if method == 'matplotlib':
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(x, y)
        # ax.set_ylim(ymin=-6, ymax=6)
        # ymax = 6
        # f[Y>ymax] = np.nan

        ax.plot_surface(X, Y, f, rstride=1, cstride=1, linewidth=0,
                        cmap='viridis', edgecolor='none')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        # ax.set_zlim(-1,1)
        # ax.set_xlim(-40, 100)

        fig.savefig(fname)
        return ax

    elif method == 'mayavi':

        from mayavi import mlab

        X, Y = np.meshgrid(x, y)
        extent = [min(x), max(x), min(y), max(y), 0, 3]
        #mlab.surf(s0 * au2ev, warp_scale="auto",extent = extent)

        if isinstance(f, list):
            for g in f:
                s = mlab.surf(g)
        else:
            s = mlab.surf(f)

        mlab.axes(s, xlabel = xlabel, ylabel = ylabel, zlabel = zlabel)

        mlab.outline(s)

        mlab.savefig(fname)
        mlab.show()

        return mlab

    else:
        raise ValueError('method has be either matplotlib or mayavi.')

def export(x, y, z, fname='output.dat', fmt='gnuplot'):
    """
    export 3D data to gnuplot format

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    fname : TYPE, optional
        DESCRIPTION. The default is 'output.dat'.
    fmt : str, optional
        The target format. The default is 'gnuplot'.

    Returns
    -------
    None.

    """
    f = open(fname, 'w')
    nx, ny = z.shape
    for j in range(ny):
        for i in range(nx):
            f.write('{} {} {}\n'.format(x[i,j], y[i,j], z[i,j]))
        f.write('\n')
    return

def plot_surface(x, y, surface):

    #data = [go.Surface(z=apes)]
    #fig = go.Figure(data = data)
    import matplotlib.pyplot as plt
    from pyqed.units import au2ev

    fig = plt.figure(figsize=(5,4))

    ax = fig.add_subplot(111, projection='3d')



    X, Y = np.meshgrid(x, y)

    ax.plot_surface(X, Y, surface * au2ev, rstride=1, cstride=1, cmap='viridis',\
                edgecolor='k',
                linewidth=0.1)

    #surf(ground)
#    ax.plot_surface(X, Y, apes1 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)
#
#    ax.plot_surface(X, Y, apes2 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)

    ax.view_init(10, -60)
    # ax.set_zlim(0, 7)
    ax.set_xlabel(r'Couping mode')
    ax.set_ylabel(r'Tuning mode')

    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Energy (eV)', rotation=90)

    #fig.subplots_adjust(top=0.95, bottom=0.16,left=0.16, right=0.9)

    plt.savefig('apes_3d.pdf')

    plt.show()
    return

def plot_surfaces(x, y, surfaces):

    #data = [go.Surface(z=apes)]
    #fig = go.Figure(data = data)
    import matplotlib.pyplot as plt
    from pyqed.units import au2ev

    fig = plt.figure(figsize=(5,4))

    ax = fig.add_subplot(111, projection='3d')



    X, Y = np.meshgrid(x, y)

    for surface in surfaces:
        ax.plot_surface(X, Y, surface * au2ev, rstride=1, cstride=1, cmap='viridis',\
                edgecolor='k',
                linewidth=0.1)

    #surf(ground)
#    ax.plot_surface(X, Y, apes1 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)
#
#    ax.plot_surface(X, Y, apes2 * au2ev, rstride=6, cstride=6, cmap='viridis', edgecolor='k'\
#                    , linewidth=0.5)

    ax.view_init(10, -60)
    # ax.set_zlim(0, 7)
    ax.set_xlabel(r'Couping mode')
    ax.set_ylabel(r'Tuning mode')

    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel('Energy (eV)', rotation=90)

    #fig.subplots_adjust(top=0.95, bottom=0.16,left=0.16, right=0.9)

    plt.savefig('apes_3d.pdf')

    plt.show()
    return

def tocolor(x, vmin=0, vmax=1):
    """
    map values to colors

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        rbga color.

    """
    from matplotlib import cm, colors
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    return mapper.to_rgba(x)

############
# tests
###########
def test_level_scheme():
    x = np.linspace(0,10)
    level_scheme(x)

    return

def vector_field(f, **kwargs):
    """
    3D plot of a vector field.

    Parameters
    ----------
    f : list of x,y,z components
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from mayavi.mlab import quiver3d
    u, v, w = f
    quiver3d(u.real, v.real, w.real, **kwargs)
    return

def scatter(points):
    n = len(points)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    return ax


if __name__ == '__main__':
    # fig, ax = subplots(ncols=1, nrows=2)
    # import numpy as np


    # test 3d
    x = np.linspace(-2,2)
    y = np.linspace(-8,8)

    X, Y = np.meshgrid(x, y)
    f = np.exp(-X**2)*np.sin(Y)
    print(f.shape)
    # ax = surf(f, x, y)

    export(X, Y, f)




