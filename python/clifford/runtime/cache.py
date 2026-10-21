import hashlib
import os
import json
import uuid

class CacheManager:
    def __init__(self, key):
        self.key = key
        cache_env = os.environ.get("XDG_CACHE_HOME")
        self.cache_path = os.path.join(f"{cache_env}/cliffordir/", key)
    
    def create_path(self, filename):
        return os.path.join(self.cache_path, filename)

    def store_group(self, data, filename):

        grp_contents = json.dumps({"child_paths" : data})
        grp_filename = "__grp__" + filename
        self.store(grp_contents, grp_filename)

    def get_group(self, filename):
        group_path = self.create_path("__grp__" + filename)
        if os.path.exists(group_path):

            try:
                with open(group_path, 'r') as f:
                    group = json.load(f)
                    return group
            except Exception as e:
                print(e)
                
        return None
    
    def store(self, data, filename):
        ext = filename.split(".")[-1]
        # print(ext)
        if ext in ("cliff", "clg"):
            data = data.str()

        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)

        
        file_path = self.create_path(filename)
        rid = str(uuid.uuid4())

        tmp_dir = os.path.join(self.cache_path, f"tmp.{rid}")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, filename)
        with open(tmp_path, "w") as f:
            f.write(data)

        os.replace(tmp_path, file_path)
        os.removedirs(tmp_dir)

        return file_path


def get_clifford_files_keys():
    keys = []

    SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # libs
    clifford_lib_hash = hashlib.sha256()
    clifford_lib = os.path.join(SRC_DIR, "_C", "libclifford.so")
    with open(clifford_lib, "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            clifford_lib_hash.update(chunk)

    keys.append(clifford_lib_hash.hexdigest())
    return "-".join(keys) 



def get_cache_key(ast_struct):
    clifford_keys = get_clifford_files_keys()
    return f"{clifford_keys}-{ast_struct.hash()}"