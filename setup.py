import itertools
from distutils.core import setup

extras_require = {
    'example': [
        'einops',
        'sacred @ git+http://github.com/IDSIA/sacred',  # The PyPi version is not frequently updated
    ],
    'test': [
        'pytest'
    ],
}
extras_require['all'] = list(set(itertools.chain.from_iterable(
    extras_require.values())
))

setup(
    name='graph_pit',
    version='0.1',
    packages=['graph_pit'],
    install_requires=[
        'numpy',
        'torch',
        'cached_property',
        'paderbox @ git+http://github.com/fgnt/paderbox',
        'padertorch @ git+http://github.com/fgnt/padertorch',
    ],
    extras_require={
        'example': [
            'einops',
            'sacred @ git+http://github.com/IDSIA/sacred',  # The PyPi version is not frequently updated
        ],
        'test': [
            'pytest'
        ],
    },
    url='https://github.com/fgnt/graph_pit',
    license='', # TODO
    author='Thilo von Neumann',
    author_email='vonneumann@nt.upb.de',
    description='PyTorch implementation of the Graph-PIT objective for '
                'training networks for continuous source separation',
)
