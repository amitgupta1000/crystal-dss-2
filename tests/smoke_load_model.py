import pickle
import sys
from google.auth import default
from google.cloud import storage

BUCKET = 'crystal-dss'
PREFIX = 'models/arima/'

creds, _ = default()
client = storage.Client(credentials=creds)

print('Listing blobs with prefix', PREFIX)
blobs = list(client.list_blobs(BUCKET, prefix=PREFIX))
if not blobs:
    print('No blobs found under prefix', PREFIX)
    sys.exit(0)

# find first .pkl
blob = None
for b in blobs:
    if b.name.endswith('.pkl'):
        blob = b
        break

if blob is None:
    print('No .pkl blob found under', PREFIX)
    sys.exit(0)

print('Using blob:', blob.name)

try:
    data = blob.download_as_bytes()
    obj = pickle.loads(data)
    print('Loaded object type:', type(obj))
except Exception as e:
    print('Failed to download or unpickle blob:', e)
    sys.exit(1)

# Try forecast/predict
tried = False
for method in ('forecast', 'predict'):
    if hasattr(obj, method):
        tried = True
        fn = getattr(obj, method)
        try:
            print(f'Calling {method}(10) ...')
            res = fn(10)
            # Show a brief summary
            print('Result type:', type(res))
            try:
                from numpy import array
                if hasattr(res, '__len__'):
                    print('First values:', list(res)[:5])
                else:
                    print('Result (repr):', repr(res)[:200])
            except Exception:
                print('Result repr:', repr(res)[:200])
        except Exception as e:
            print(f'{method} raised an exception:', e)
        break

if not tried:
    print('No forecast/predict method found on loaded object.')

print('Smoke test complete.')
