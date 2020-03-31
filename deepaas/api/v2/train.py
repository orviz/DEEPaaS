# -*- coding: utf-8 -*-

# Copyright 2018 Spanish National Research Council (CSIC)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import asyncio
from datetime import datetime
import functools
import json
import os
import pathlib
import uuid

from aiohttp import web
import aiohttp_apispec
from oslo_log import log
from webargs import aiohttpparser
import webargs.core

from deepaas.api.v2 import responses
from deepaas.api.v2 import utils
from deepaas import model

LOG = log.getLogger("deepaas.api.v2.train")

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))

IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
        LOG.info('IN_OUT_BASE_DIR=%s' % IN_OUT_BASE_DIR)
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR=" + \
        "{}\" is not a valid directory! ".format(env_in_out_base_dir) + \
        "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        LOG.info(msg)

DEEPAAS_CACHE_DIR = os.path.join(IN_OUT_BASE_DIR, 'cache', 'deepaas')
if os.path.exists(DEEPAAS_CACHE_DIR):
    if not os.path.isdir(DEEPAAS_CACHE_DIR):
        LOG.info('ERROR: %s is not a directory!' % DEEPAAS_CACHE_DIR)
else:
    LOG.info('creating %s ...' % DEEPAAS_CACHE_DIR)
    try:
        pathlib.Path(DEEPAAS_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        LOG.info('creating %s ... %s' % (DEEPAAS_CACHE_DIR, 'OK' if os.path.isdir(DEEPAAS_CACHE_DIR) else 'ERROR'))
    except Exception as e:
        LOG.info('creating %s ... ERROR: %s' % (DEEPAAS_CACHE_DIR, e))
DEEPAAS_TRAINING_HISTORY_FILE = os.path.join(DEEPAAS_CACHE_DIR, 'training_history.json')

def load_training_history(file):
    ret = {}
    if os.path.isfile(file):
        with open(file, 'rb') as fp:
            data = fp.read()
            if data:
                ret = json.loads(data.decode('utf-8'))
    for uuid in ret.keys():
        try:
            # unfinished tasks due to an improper system shutdown
            if ret[uuid]["task"]["status"]["running"]:
                ret[uuid]["task"]["status"]["running"] = "error"
        except KeyError:
            LOG.info('cannot read task %s or its status from the training history file %s' % (uuid, file))
    return ret

def save_training_history(trainings, file):
    with open(file, 'wb') as fp:
        try:
            data = json.dumps(trainings, indent=4)
            data = bytes(data, 'utf-8')
            fp.write(data)
        except Exception as e:
            LOG.info(e)

def _get_handler(model_name, model_obj):  # noqa
    args = webargs.core.dict2schema(model_obj.get_train_args())
    args.opts.ordered = True

    class Handler(object):
        model_name = None
        model_obj = None

        def __init__(self, model_name, model_obj):
            self.model_name = model_name
            self.model_obj = model_obj
            self._trainings = {}
            self._training_history_file = DEEPAAS_TRAINING_HISTORY_FILE
            self._training_history = load_training_history(self._training_history_file)

        @staticmethod
        def build_train_response(uuid, training):
            if not training:
                return

            ret = {}
            ret["date"] = training["date"]
            ret["args"] = training["args"]
            ret["uuid"] = uuid

            if training["task"].cancelled():
                ret["status"] = "cancelled"
            elif training["task"].done():
                exc = training["task"].exception()
                if exc:
                    ret["status"] = "error"
                    ret["message"] = "%s" % exc
                else:
                    ret["status"] = "done"
                    ret["result"] = training["task"].result()
                    end = datetime.strptime(ret["result"]["finish_date"],
                                            '%Y-%m-%d %H:%M:%S.%f')
                    start = datetime.strptime(ret["date"],
                                              '%Y-%m-%d %H:%M:%S.%f')
                    ret["result"]["duration"] = str(end - start)
            else:
                ret["status"] = "running"
            return ret

        def _train_task_finished_callback(self, uuid_, result):
            ret = self.build_train_response(uuid_, self._trainings[uuid_])
            self._training_history[uuid_] = ret
            save_training_history(self._training_history, self._training_history_file)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Retrain model with available data"
        )
        @aiohttp_apispec.querystring_schema(args)
        @aiohttpparser.parser.use_args(args)
        async def post(self, request, args, wsk_args=None):
            uuid_ = uuid.uuid4().hex
            train_task = self.model_obj.train(**args)
            train_task.add_done_callback(functools.partial(self._train_task_finished_callback, uuid_))
            self._trainings[uuid_] = {
                "date": str(datetime.now()),
                "task": train_task,
                "args": args,
            }
            ret = self.build_train_response(uuid_, self._trainings[uuid_])
            self._training_history[uuid_] = ret
            save_training_history(self._training_history, self._training_history_file)
            return web.json_response(ret)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Cancel a running training"
        )
        async def delete(self, request, wsk_args=None):
            uuid_ = request.match_info["uuid"]
            training = self._trainings.pop(uuid_, None)
            if not training:
                raise web.HTTPNotFound()
            training["task"].cancel()
            try:
                await asyncio.wait_for(training["task"], 5)
            except asyncio.TimeoutError:
                pass
            LOG.info("Training %s has been cancelled" % uuid_)
            ret = self.build_train_response(uuid_, training)
            self._training_history[uuid_] = ret
            save_training_history(self._training_history, self._training_history_file)
            return web.json_response(ret)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Get a list of trainings (running or completed)"
        )
        @aiohttp_apispec.response_schema(responses.TrainingList(), 200)
        async def index(self, request, wsk_args=None):
            ret = []
            for uuid_, training in self._trainings.items():
                training = self._trainings.get(uuid_, None)
                aux = self.build_train_response(uuid_, training)
                ret.append(aux)
                self._training_history[uuid_] = ret
            save_training_history(self._training_history, self._training_history_file)
            return web.json_response(ret)

        @aiohttp_apispec.docs(
            tags=["models"],
            summary="Get status of a training"
        )
        @aiohttp_apispec.response_schema(responses.Training(), 200)
        async def get(self, request, wsk_args=None):
            uuid_ = request.match_info["uuid"]
            training = self._trainings.get(uuid_, None)
            ret = self.build_train_response(uuid_, training)
            if ret:
                self._training_history[uuid_] = ret
                save_training_history(self._training_history, self._training_history_file)
                return web.json_response(ret)
            raise web.HTTPNotFound()

    return Handler(model_name, model_obj)


def setup_routes(app, enable=True):
    # In the next lines we iterate over the loaded models and create the
    # different resources for each model. This way we can also load the
    # expected parameters if needed (as in the training method).
    for model_name, model_obj in model.V2_MODELS.items():
        if enable:
            hdlr = _get_handler(model_name, model_obj)
        else:
            hdlr = utils.NotEnabledHandler()
        app.router.add_post(
            "/models/%s/train/" % model_name,
            hdlr.post
        )
        app.router.add_get(
            "/models/%s/train/" % model_name,
            hdlr.index,
            allow_head=False
        )
        app.router.add_get(
            "/models/%s/train/{uuid}" % model_name,
            hdlr.get,
            allow_head=False
        )
        app.router.add_delete(
            "/models/%s/train/{uuid}" % model_name,
            hdlr.delete
        )
