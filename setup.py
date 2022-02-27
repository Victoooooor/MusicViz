from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'MusicViz'
LONG_DESCRIPTION = 'Music visualization using CNN predicting Valence Arousal -> biggan -> style Transform'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="MusicViz",
    version=VERSION,
    author="Victor Zhang",
    author_email="<victorzh716@email.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'imgaug',
        'tensorflow',
        'matplotlib',
        'pandas',
        'tqdm',
        'Pillow',
        'progressbar',
        'IPython',
        'opencv-python',
        'keras',
        'python-magic'
        'pytorch_pretrained_biggan',
        'scikit_learn',
        'scipy',
        'umap_learn'
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python', 'tensorflow', 'Music', 'Music visualization', 'Biggan', 'Style Transfer', 'GAN'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux :: ArchLinux",
        "Operating System :: Microsoft :: Windows",
    ]
)