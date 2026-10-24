from .import_packages import *


class WriterLoger(object):
    def __init__(self, loger_dir: str, loger_name: str):
        self.json_path = os.path.join(loger_dir, loger_name)
        with open(self.json_path, "w") as f:
            f.close()

    def __call__(self, item: dict):
        with open(self.json_path, "a+") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
