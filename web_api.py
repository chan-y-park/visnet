import time
import flask
from io import BytesIO


class VisNetWebApp(flask.Flask):
    def __init__(self):
        super().__init__('VisNet')
        self._input_image = None 

    def load_image_from_flask(self, file_storage):
        self._input_image = BytesIO()
        file_storage.save(self._input_image)
        self._input_image.seek(0)


def get_web_app():
    web_app = VisNetWebApp()
    web_app.config.update(
        DEBUG=True,
        SECRET_KEY='visnet secret key',
    )
    web_app.add_url_rule(
        '/', 'index', index, methods=['GET'],
    )
    web_app.add_url_rule(
        '/config', 'config', config, methods=['GET', 'POST'],
    )
    web_app.add_url_rule(
        '/input_image', 'input_image', input_image, methods=['GET'],
    )
    web_app.add_url_rule(
        '/results', 'show_results', show_results, methods=['POST'],
    )

    return web_app


def index():
    #return flask.render_template('index.html')
    return flask.redirect(flask.url_for('config'))


def config():
    app = flask.current_app
    image_file = None
    if flask.request.method == 'POST':
        try:
            image_file = flask.request.files['image_file']
        except KeyError:
            pass

    if image_file is not None and image_file.filename != '':
        app.load_image_from_flask(image_file) 
        show_input_image = True
    else:
        # TODO: Load a default image file.
        show_input_image = False

    return flask.render_template(
        'config.html',
        show_input_image=str(show_input_image),
    )


def show_results():
    pass

def input_image():
    input_image = flask.current_app._input_image
    if input_image is not None:
        rv = flask.send_file(
            input_image,
            mimetype='image/jpg',
            cache_timeout=0,
            as_attachment=False,
            attachment_filename='input_image.jpg',
        )
        rv.set_etag(str(time.time()))
        return rv
