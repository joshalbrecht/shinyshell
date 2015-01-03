
"""
This is the highest level module. Nothing except the root folder should import from here
(ie, things in optics should not import from here.)

Why?

Because these object presuppose a window, user, etc, whereas optics can run in headless mode.
"""