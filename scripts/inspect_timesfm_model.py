from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams

try:
    hparams = TimesFmHparams(backend="cpu", horizon_len=10)
    checkpoint = TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch")
    tfm = TimesFm(hparams=hparams, checkpoint=checkpoint)
    print("SUCCESS: tfm instantiated")
    try:
        print(type(tfm._model))
    except Exception as e:
        print("Could not access tfm._model:", e)
except Exception as exc:
    print("ERROR_INSTANCIATING_TIMESFM:")
    import traceback

    traceback.print_exc()
