import os

# closeai的key
os.environ['OPENAI_API_KEY'] = 'sk-O1ciMpnE1yOK21eW7o1tV2JskNUMeZxlPDKfeJUa5hC12rnw'
# azure gpt-4的key
# os.environ['OPENAI_API_KEY'] = 'e8f5fd1cfbab4335849a530b8b42e596'

from .main import *  # noqa

__version__ = "0.3.1"
