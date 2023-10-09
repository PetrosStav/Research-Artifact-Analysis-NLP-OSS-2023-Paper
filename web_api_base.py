# --------------------------------------------------------- #
# imports
# from importlib.metadata import requires
import traceback
import logging

# --------------------------------------------------------- #
# from imports
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from waitress import serve
from ast import literal_eval

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
m_name = 'Flan-T5-Base'

from flan_t5_base_api import infer_text, infer_text_web, tokenize_text

print('Loaded Model!')

logging.basicConfig(filename='./logs/transformers_web_api.log', level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s")

app = Flask(__name__)

@app.route('/web_api', methods=['GET', 'POST'])
def web_api():
    try:
        input = ''
        output = ''
        args = ''
        max_len = 100
        if request.method == 'POST':
            input = request.form.get('input')
            max_len = int(request.form.get('maxlen'))
            args = request.form.get('args')
            if not args.strip():
                args_dict = {}
            else:
                args_dict = {x.split('=')[0].strip(): x.split('=')[1].strip() for x in args.split('\n')}
                args_dict = {x: literal_eval(args_dict[x]) for x in args_dict}
            output = infer_text_web(input, max_len=max_len, **args_dict)
        return render_template('web_api.html', data0=m_name, data1=input, data2=output, data3=max_len, data4=args)
    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e) + '\n' + traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)


# --------------------------------------------------------- #

@app.route('/infer_sentence', methods=['GET', 'POST'])
def infer_sentence():
    try:
        app.logger.debug("JSON received...")
        app.logger.debug(request.json)
        if request.json:
            mydata = request.json

            res = infer_text(mydata['text'], **mydata['gen_config'])

            res = {
                'input': mydata['text'],
                'output': res
            }

            return jsonify(res)
        else:
            ret = {'success': 0, 'message': 'request should be json formatted'}
            app.logger.debug(ret)
            return jsonify(ret)

    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e) + '\n' + traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)


@app.route('/tokenize_sentence', methods=['GET', 'POST'])
def tokenize_sentence():
    try:
        app.logger.debug("JSON received...")
        app.logger.debug(request.json)
        if request.json:
            mydata = request.json

            res = tokenize_text(mydata['text'])

            res = {
                'input': mydata['text'],
                'output': res
            }

            return jsonify(res)
        else:
            ret = {'success': 0, 'message': 'request should be json formatted'}
            app.logger.debug(ret)
            return jsonify(ret)

    except Exception as e:
        app.logger.debug(str(e))
        traceback.print_exc()
        ret = {'success': 0, 'message': str(e) + '\n' + traceback.format_exc()}
        app.logger.debug(ret)
        return jsonify(ret)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=9001)
