from distutils.core import setup
import itertools

extras = {
    'example': [
        'einops',
        'lazy_dataset',
        # The PyPi version is not frequently updated
        'sacred @ git+http://github.com/IDSIA/sacred',
        'padertorch @ git+http://github.com/fgnt/padertorch',
        'paderbox @ git+http://github.com/fgnt/paderbox',
    ],
    'test': [
        'pytest',
        'paderbox @ git+http://github.com/fgnt/paderbox',
    ],
}
extras['all'] = list(set(itertools.chain.from_iterable(extras.values())))

setup(
    name='graph_pit',
    version='0.1',
    packages=['graph_pit'],
    install_requires=[
        'numpy',
        'torch',
        'cached_property',
    ],
    extras_require=extras,
    url='https://github.com/fgnt/graph_pit',
    license='',  # TODO
    author='Thilo von Neumann',
    author_email='',
    description='PyTorch implementation of the Graph-PIT objective for '
                'training networks for continuous source separation',
)
