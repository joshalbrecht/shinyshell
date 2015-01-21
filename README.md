shinyshell
==========

Python scripts for creating freeform optical surfaces

Conventions:

 - Everything with mu and rho should be mu first, then rho
 - All imports are of the form:
import optics.globals
EXCEPT:
from optics.base import * # pylint: disable=W0401,W0614
which is the one exception. Otherwise it is REALLY annoying to refactor python code.
