#! /usr/bin/env python
import sys
import getopt

from web_api import get_web_app

application = get_web_app()

if __name__ == '__main__':
    host = '0.0.0.0'
    port = '9999'
    opts, args = getopt.getopt(sys.argv[1:], 'p:')
    for opt, arg in opts:
        if opt == '-p':
            port = int(arg)

    web_app.run(
        host=host,
        port=port,
        debug=True,
        use_reloader=False,
        threaded=True,
    )
