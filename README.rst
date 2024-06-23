========
citsigol
========

Citsigol is an investigatory package providing utilities for investigation and manipulation of the reversed logistic (citsigol) map.

* Free software: MIT license

..
    * Documentation: (Will eventually be at) https://citsigol.readthedocs.io.

Features
--------

* Provides the logistic map and its inverse, the citsigol map.
* Provides general utilities for investigating other maps.
* Bifurcation diagrams include dynamic zooming and generally retain plot quality up to machine precision limitations.
* Sequence generators

Imagine this... But *backwards!*

.. image:: https://lh3.googleusercontent.com/d/1pgT04PvnFX8Mz53zffG_CGOcUdUhamEP


The citsigol (reverse logistic) map is a strange thing...

.. image:: https://lh3.googleusercontent.com/d/1bIm20r7oPoGTdU4NXLBt157TllT0E_pC


It has a fingerprint-like structure...

.. image:: https://lh3.googleusercontent.com/d/1eW0LlWZnE9CVI10xFG-vTUk0yDcbxrPn

The citsigol map branches on each iteration, which causes some interesting challenges for working with it.

.. image:: https://lh3.googleusercontent.com/d/1c_1zBR_PDuo6cSnXD_hCKQC7bwSHUI4W

Gratuitous high zoom logistic map bifurcation diagram plot...

.. image:: https://lh3.googleusercontent.com/d/1OreqbaS26oc34hXWPTH-tMEofyiiRmx0

Also included are tools to investigate arbitrary maps, such as this one:

.. image:: https://lh3.googleusercontent.com/d/1Fj6e2Uj35NQhh9XxaTBVt9eiB8AtCM47


..
    .. image:: https://img.shields.io/pypi/v/citsigol.svg
            :target: https://pypi.python.org/pypi/citsigol
    
    .. image:: https://img.shields.io/travis/7c-c7/citsigol.svg
            :target: https://travis-ci.com/7c-c7/citsigol
    
    .. image:: https://readthedocs.org/projects/citsigol/badge/?version=latest
            :target: https://citsigol.readthedocs.io/en/latest/?version=latest
            :alt: Documentation Status


Try it out!
-----------

Clone the repo and install it from source into your system's python installation:

::

    git clone https://github.com/7c-c7/citsigol.git
    pip install ./citsigol
    python ./citsigol/src/citsigol/demo.py

In the matplotlib window that opens, you'll see a logistic bifurcation diagram. You can click and drag to zoom. The diagram should repopulate itself dynamically.

If you have any issues, please report them! This is a learning effort, and we are happy to get feedback and improve the package.

Development
-----------
Setup should be quite standard. The following should be sufficient to get started:

::

    git clone https://github.com/7c-c7/citsigol.git
    cd citsigol
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements_dev.txt
    pip install -e .

You can test the package by running `pytest` in the root directory.

This package was built with Python 3.12 and later in mind, It may work with earlier versions but this is not guaranteed.

We would be happy to review your Pull Requests and to see your ideas for features and improvements.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
