"""Single source of truth for ATHENA-R1's version string.

Lives in its own module so ``python -m athena_r1 --help`` can read it
without triggering the heavy ``transformers`` / ``tooluniverse`` import
chain that ``athena_r1/__init__.py`` pulls in.
"""

__version__ = "1.0.0"
