import json
import queue
import threading
import time


class PrefetchingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rules = self._get_prefetch_rule(kwargs['prefetch_rule_file'])
        self.queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def load_weights(self, module, *_):
        # Do prefetching
        targets = self.rules.get(module, [])
        for tgt_module in targets:
            tgt_module.loading = True
            self.queue.put(tgt_module)

        self._wait_weights(module)

    def _do_load_and_setflag(self, module):
        super().load_weights(module)
        module.loaded = True

    def release_weights(self, module, *args, **kwargs):
        super().release_weights(module, *args, **kwargs)
        module.loaded = False
        module.loading = False

    def _wait_weights(self, module):
        # if module.loaded:
        #     print('Success prefetch! no need to wait')
        # else:
        #     print('No prefetch!')
        if not module.loading:
            print(f'Warning, not loading {module.param_path}')
            self._do_load_and_setflag(module)
            return
        while not module.loaded:
            if not module.loading:
                raise AssertionError
            time.sleep(0.000001)

    def _worker(self):
        while True:  # TODO when to terminate
            module = self.queue.get()
            self._do_load_and_setflag(module)
            self.queue.task_done()

    def _get_prefetch_rule(self, path):
        name2module = dict(self.model.named_modules())
        with open(path) as f:
            return {
                name2module[key]: [name2module[v] for v in value]
                for key, value in json.load(f).items()
            }
