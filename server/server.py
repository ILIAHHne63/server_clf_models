import sys
import inspect
import traceback
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor

import uvicorn
import os
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import create_model

from classifer import TextClassifier
from structures import Labels, Prediction, Scores
from structures import FitConfig, PredictConfig, Texts, ReturnValue, PredictReturnValue, ModelConfig
from os import environ as env

MODEL_DIR_LOADED = env['MODEL_DIR_LOADED']
MODEL_DIR = env['MODEL_DIR']
AVAILABLE_CORES = multiprocessing.Value('d', int(env['AVAILABLE_CORES']))
MAX_MODELS = int(env['MAX_MODELS'])
LOADED_MODELS = {}
LOCK = multiprocessing.Lock()

app = FastAPI()
def get_params(method):
    return {k: (v.annotation, ...) for k, v in inspect.signature(method).parameters.items()}

@app.post("/fit", response_model=ReturnValue, name='Fit')
async def fit(request: create_model('FitInput', **get_params(TextClassifier.fit))):

    global AVAILABLE_CORES

    try:
        with LOCK:
            print(AVAILABLE_CORES.value)
            AVAILABLE_CORES.value -= 1
            if AVAILABLE_CORES.value < 0:
                raise HTTPException(status_code=400,
                                    detail='Lack of available cores')

        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor() as pool:
            await loop.run_in_executor(pool, TextClassifier.fit, request.texts, request.labels, request.config)

        with LOCK:
            print(AVAILABLE_CORES.value)
            AVAILABLE_CORES.value += 1

        return ReturnValue(success=True, message='Fitting',
                           traceback='Everything is OK')

    except Exception as error:
        with LOCK:
            print(AVAILABLE_CORES.value)
            AVAILABLE_CORES.value += 1

        return ReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )



@app.post("/predict", response_model=PredictReturnValue, name='Predict')
async def predict(request: create_model('PredictInput', **get_params(TextClassifier.predict))):
    try:
        if request.config.model_name not in LOADED_MODELS:
            raise HTTPException(status_code=400, detail='Model not loaded')

        return PredictReturnValue(success=True, message="Predictions", traceback="Everything is Ok",
                                  prediction=TextClassifier.predict(texts=request.texts, config=request.config))
    except Exception as error:
        return PredictReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
            prediction=None,
        )
#
@app.post("/load", name='Load')
async def load_model(request: ModelConfig):

    root_path = Path(MODEL_DIR_LOADED)
    if not Path(MODEL_DIR_LOADED).exists():
        root_path.mkdir()

    try:

        if request.model_name in LOADED_MODELS:
            return ReturnValue(success=True, message='Model already loaded', traceback='Everything is OK')

        if len(LOADED_MODELS) >= MAX_MODELS:
            raise HTTPException(status_code=400,
                                detail='Maximum number of models reached')

        if not os.path.exists(os.path.join(MODEL_DIR, request.model_name)):
            raise HTTPException(status_code=400,
                                detail='Model does not exist')

        print(len(LOADED_MODELS))
        LOADED_MODELS[request.model_name] = True
        shutil.copytree(os.path.join(MODEL_DIR, request.model_name), os.path.join(MODEL_DIR_LOADED, request.model_name))
        print(len(LOADED_MODELS))

        return ReturnValue(success=True, message='Model loaded', traceback='Everything is OK')

    except Exception as error:
        return ReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )

@app.post("/unload", name='Unload')
async def unload_model(request: ModelConfig):

    try:

        if request.model_name not in LOADED_MODELS:
            raise HTTPException(status_code=400,
                                detail='Model not loaded')

        print(len(LOADED_MODELS))
        del LOADED_MODELS[request.model_name]
        shutil.rmtree(os.path.join(MODEL_DIR_LOADED, request.model_name))

        print(len(LOADED_MODELS))
        return ReturnValue(success=True, message='Model unloaded', traceback='Everything is OK')

    except Exception as error:
        return ReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )

@app.post("/remove", name='Remove')
async def remove_model(request: ModelConfig):
    try:

        if not os.path.exists(os.path.join(MODEL_DIR, request.model_name)):
            raise HTTPException(status_code=400,
                                detail='Model not found')

        shutil.rmtree(os.path.join(MODEL_DIR, request.model_name))
        if request.model_name in LOADED_MODELS:
            shutil.rmtree(os.path.join(MODEL_DIR_LOADED, request.model_name))
            del LOADED_MODELS[request.model_name]
        return ReturnValue(success=True, message='Model are removed', traceback='Everything is OK')

    except Exception as error:
        return ReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )


@app.post("/remove_all", name='RemoveAll')
async def remove_all_models():
    try:
        for filename in os.listdir(MODEL_DIR):
            file_path = os.path.join(MODEL_DIR, filename)
            shutil.rmtree(file_path)

        for filename in os.listdir(MODEL_DIR_LOADED):
            file_path = os.path.join(MODEL_DIR_LOADED, filename)
            shutil.rmtree(file_path)

        LOADED_MODELS.clear()

        return ReturnValue(success=True, message='All models are removed', traceback='Everything is OK')

    except Exception as error:
        return ReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )

if __name__ == '__main__':
     if len(sys.argv) != 3:
         print('Run `python server.py <HOST> <PORT>`')
         sys.exit(1)

     host = sys.argv[1]
     port = int(sys.argv[2])

     uvicorn.run('sever:app', host=host, port=port)
