import time
import uuid
import flask
import PIL
from io import BytesIO

from api import InputImage
from api import get_all_deconv_results
from api import get_deconv_images

class VisNetWebApp(flask.Flask):
    def __init__(self):
        super().__init__('VisNet')
        self.use_cpu = False

    def load_image_from_flask(self, file_storage):
        image_id = str(uuid.uuid4())
        # TODO: Check image type.
        with open('deconv_results/{}.jpg'.format(image_id), 'wb') as fp:
            file_storage.save(fp)
        return image_id


def get_web_app(
    use_cpu=False,
    full_deconv=True,    
):
    web_app = VisNetWebApp()
    web_app.config.update(
        DEBUG=True,
        SECRET_KEY='visnet secret key',
    )

    web_app.use_cpu = use_cpu
    web_app.full_deconv = full_deconv

    web_app.add_url_rule(
        '/', 'index', index, methods=['GET'],
    )
    web_app.add_url_rule(
        '/config', 'config', config, methods=['GET', 'POST'],
    )
    web_app.add_url_rule(
        '/input_image/<image_id>', 'input_image', input_image, methods=['GET'],
    )
    web_app.add_url_rule(
        '/deconv_image/<image_id>', 'deconv_image', deconv_image,
        methods=['GET'],
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
        image_id = app.load_image_from_flask(image_file) 
        return flask.render_template(
            'config.html',
            image_id=image_id,
        )
    else:
        # TODO: Load a default image file.
        return flask.render_template('config.html')


def show_results():
    app = flask.current_app
    image_id = flask.request.form['image_id']
    num_top_features=3
    input_image_path = get_image_path(image_id, 'jpg')
    input_image = InputImage(input_image_path)
    input_image = input_image.get_resized_image(224)
    input_image.image.save(input_image_path)

    rd = get_all_deconv_results(
        input_image,
        log_device_placement=False,
        num_top_features=num_top_features,
        use_cpu=app.use_cpu,
        full_deconv=app.full_deconv,
    ) 

    get_deconv_images(
        input_image,
        save_path=get_image_path(image_id, 'svg'),
        num_top_features=num_top_features,
        deconv_layers=rd['deconv_layers'],
    )
    
    return flask.render_template(
        'deconv_result.html',
        image_id=image_id,
    )

def input_image(image_id):
    try:
        rv = flask.send_file(
            get_image_path(image_id, 'jpg'),
            mimetype='image/jpg',
            cache_timeout=0,
            as_attachment=False,
            attachment_filename='input_image.jpg',
        )
        rv.set_etag(str(time.time()))
        return rv
    except:
        # TODO: Check exception type & gracefully return.
        raise RuntimeError('No input image file.')


def deconv_image(image_id):
    try:
        rv = flask.send_file(
            get_image_path(image_id, 'svg'),
            mimetype='image/svg+xml',
            cache_timeout=0,
            as_attachment=False,
            attachment_filename='deconv_image.svg',
        )
        rv.set_etag(str(time.time()))
        return rv
    except:
        # TODO: Check exception type & gracefully return.
        raise RuntimeError('No deconv result.')


def get_image_path(image_id, ext='jpg'):
    return 'deconv_results/{}.{}'.format(image_id, ext)
