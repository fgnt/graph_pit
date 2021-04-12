from distutils.core import setup

setup(
    name='graph_pit',
    version='0.1',
    packages=['graph_pit'],
    install_requires=['torch'],
    url='',
    license='', # TODO
    author='thequilo',
    author_email='vonneumann@nt.upb.de',
    description='PyTorch implementation of the Graph-PIT objective for '
                'training networks for continuous source separation'
)
