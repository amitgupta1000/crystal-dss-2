import traceback
from google.cloud import storage
from google.auth import default
import pickle

try:
    creds, _ = default()
    client = storage.Client(credentials=creds)
    bucket = client.bucket('crystal-dss')
    prefix = 'models/arima/'
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"Found {len(blobs)} blobs under {prefix}")
    for b in blobs:
        print(f"- {b.name} (size={b.size})")

    # attempt to download first .pkl
    pkl_blobs = [b for b in blobs if b.name.endswith('.pkl')]
    if not pkl_blobs:
        print('No .pkl blobs to test.')
    else:
        b = pkl_blobs[0]
        print(f"Downloading {b.name}...")
        try:
            data = b.download_as_bytes()
            print(f"Downloaded {len(data)} bytes")
            try:
                obj = pickle.loads(data)
                print('Unpickle succeeded. Type:', type(obj))
                # attempt to print some attributes if available
                for attr in ('order','seasonal_order','aic'):
                    try:
                        val = getattr(obj, attr)
                        if callable(val):
                            val = val()
                        print(f"  {attr}: {val}")
                    except Exception as e:
                        print(f"  Could not read {attr}: {e}")
            except Exception as e:
                print('Unpickle failed:')
                traceback.print_exc()
        except Exception:
            print('Download failed:')
            traceback.print_exc()
except Exception:
    print('Top-level failure:')
    traceback.print_exc()
