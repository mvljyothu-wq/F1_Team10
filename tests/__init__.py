import logging

# Keep test output readable: the receiver logs expected rejections as warnings.
logging.getLogger("rudp").addHandler(logging.NullHandler())
logging.getLogger("rudp").propagate = False
